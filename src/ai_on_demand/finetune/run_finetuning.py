import torch
import cv2
from .finetuning_utils import SingleClassInstanceDataset
from .finetuning_utils import PanopticLoss
from torch.utils.data import DataLoader
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
    epochs = config.get("epochs") or 5
    batch_size = config.get("batch_size") or 16

    # allows falsy values - "", None
    data_cls = SingleClassInstanceDataset
    train_dataset = data_cls(
        train_dir, transforms=tfs, weight_gamma=0.7
    )  # what does weight gamma mean?

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )

    model = torch.jit.load(model_dir, map_location=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
        images = images.permute(0, 3, 1, 2).float()
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
