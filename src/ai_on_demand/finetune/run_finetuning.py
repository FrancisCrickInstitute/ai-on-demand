import torch
import cv2
from .finetuning_utils import SingleClassInstanceDataset
from .finetuning_utils import PanopticLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

# from sklearn import measure
# from skimage import io

augmentations = sorted(
    name
    for name in A.__dict__
    if callable(A.__dict__[name])
    and not name.startswith("__")
    and name[0].isupper()
)

augmentations_dict = [
    {"aug": "RandomScale", "scale_limit": [-0.9, 1]},
    {
        "aug": "PadIfNeeded",
        "min_height": 256,
        "min_width": 256,
        "border_mode": 0,
    },
    {"aug": "RandomCrop", "height": 256, "width": 256},
    {"aug": "Rotate", "limit": 180, "border_mode": 0},
    {
        "aug": "RandomBrightnessContrast",
        "brightness_limit": 0.3,
        "contrast_limit": 0.3,
    },
    {"aug": "HorizontalFlip"},
    {"aug": "VerticalFlip"},
]

# set the training image augmentations
norms = {"mean": 0.57571, "std": 0.12765}
aug_string = []
dataset_augs = []
for aug_params in augmentations_dict:
    aug_name = aug_params["aug"]

    assert (
        aug_name in augmentations or aug_name == "CopyPaste"
    ), f"{aug_name} is not a valid albumentations augmentation!"

    aug_string.append(aug_params["aug"])
    del aug_params["aug"]
    dataset_augs.append(A.__dict__[aug_name](**aug_params))

aug_string = ",".join(aug_string)

tfs = A.Compose([*dataset_augs, A.Normalize(**norms), ToTensorV2()])


def finetune(config):
    print("finetuning")
    # why not
    # device = config.get("device", "cpu")
    # if there is a device key and it is truthy
    device = config.get("device") or "cpu"

    train_dir = config["train_dir"]
    model_dir = config["model_dir"]
    save_dir = config["save_dir"]
    save_name = config["save_name"]
    epochs = config["epochs"]
    finetune_layer = config["layers"]
    batch_size = config.get("batch_size") or 16

    # allows falsy values - "", None
    data_cls = SingleClassInstanceDataset
    train_dataset = data_cls(train_dir, transforms=tfs, weight_gamma=None)

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )

    model = torch.jit.load(model_dir, map_location=device)

    # freeze all encoder layers
    for pname, param in model.named_parameters():
        if "encoder" in pname:
            param.requires_grad = False

    # freeze specific layers
    if finetune_layer == "none":
        pass
    elif finetune_layer == "all":
        for pname, param in model.named_parameters():
            if "encoder" in pname:
                param.requires_grad = True
    else:
        # unfreeze is cumulative from layer 1 to chosen layer
        layers = ["layer1", "layer2", "layer3", "layer4"]
        for layer_name in layers[layers.index(finetune_layer) :]:
            for pname, param in model.named_parameters():
                if layer_name in pname:
                    param.requires_grad = True
    num_trainable = sum(
        p[1].numel() for p in model.named_parameters() if p[1].requires_grad
    )
    print(f"Training {num_trainable} parameters.")

    optimizer = configure_optimizer(model, "AdamW", weight_decay=0.1)

    criterion = PanopticLoss()
    for epoch in range(epochs):
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )
    torch.jit.save(model, save_dir + "/" + save_name + ".pth")


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch=None,
    config=None,
    device="cpu",
):
    model.train()
    for i, batch in enumerate(train_loader):
        images = batch["image"]
        target = {
            k: v for k, v in batch.items() if k not in ["image", "fname"]
        }

        # move to gpu or cpu (aren't they arealyda on the cpu?)
        # images = images.permute(0, 3, 1, 2).float()
        images = images.float()
        images = images.to(device, non_blocking=True)
        target = {
            k: tensor.to(device, non_blocking=True)
            for k, tensor in target.items()
        }
        print(f"image shape: {images.shape}")

        optimizer.zero_grad()

        output = model(images)
        print(output.keys())
        loss, aux_loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"epoch = {epoch}")
        print(aux_loss)
        print(loss.item())


def configure_optimizer(model, opt_name, **opt_params):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.

    Only BatchNorm parameters and biases are excluded.
    """

    # easy if there's no weight_decay
    if "weight_decay" not in opt_params:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)
    elif opt_params["weight_decay"] == 0:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)

    decay = set()
    no_decay = set()
    param_dict = {}

    blacklist = (torch.nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = "%s.%s" % (mn, pn) if mn else pn

            if full_name.endswith("bias"):
                no_decay.add(full_name)
            elif full_name.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

            param_dict[full_name] = p

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "Overlapping decay and no decay"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    param_groups = [
        {"params": decay_params, **opt_params},
        {"params": no_decay_params, **opt_params},
    ]
    param_groups[1][
        "weight_decay"
    ] = 0  # overwrite default to 0 for no_decay group

    return optim.__dict__[opt_name](param_groups, **opt_params)
