from collections import namedtuple, defaultdict

from .tasks import TASK_NAMES

# Define model shorthand and display names
MODEL_NAMES = {
    "sam": "Segment Anything",
    "unet": "U-Net",
}
# Reverse dict for convenience
MODEL_DISPLAYNAMES = {v: k for k, v in MODEL_NAMES.items()}

# # Checker function so that all models have definitons below
# def check_model_names(d):
#     return all([k in MODEL_NAMES for k in d.keys()])

# # Define the available versions of each model
# MODEL_VERSIONS = {
#     "sam": ["default", "vit_l", "vit_b", "MedSAM"],
#     "unet": ["U-Net", "Attention U-Net"],
# }
# assert check_model_names(MODEL_VERSIONS)


# # Checker function so that all model tasks are valid
# def check_model_tasks(d):
#     return all([task in TASK_NAMES for tasks in d.values() for task in tasks])


# # Define the tasks each model can perform (at the meta-level)
# # Each model may have variants that can only do one task, which is fine, but
# # full capability is listed here
# MODEL_TASKS = {
#     "sam": ["everything"],
#     "unet": ["mito", "er", "ne"],
# }
# check_model_tasks(MODEL_TASKS)
# # Create the reverse dict to check available models per task
# TASK_MODELS = defaultdict(list)
# for model, tasks in MODEL_TASKS.items():
#     for task in tasks:
#         TASK_MODELS[task].append(model)

MODEL_TASK_VERSIONS = {
    "sam": {
        "everything": {
            "default": {
                "filename": "sam_default.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            },
            "vit_h": {
                "filename": "sam_vit_h.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            },
            "vit_l": {
                "filename": "sam_vit_l.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            },
            "vit_b": {
                "filename": "sam_vit_b.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            },
            "MedSAM": {
                "filename": "sam_MedSAM.pth",
                "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth",
            },
        }
    },
    "unet": {
        "mito": {
            "Attention U-Net": {
                "filename": "Attention_HUNet_3e5_Adam_restart_12_16.best.1266.pt",
                "dir": "/nemo/stp/ddt/working/shandc/aiod_models/",
            },
        },
        "ne": {
            "Attention U-Net": {
                "filename": "Attention_HUNet_NE.best.368.pt",
                "dir": "/nemo/stp/ddt/working/shandc/aiod_models/",
            }
        },
    },
}

# Check each model name exists
assert all([k in MODEL_NAMES for k in MODEL_TASK_VERSIONS.keys()])
# Check each task exists
assert all(
    [
        k in TASK_NAMES
        for model_d in MODEL_TASK_VERSIONS.values()
        for k in model_d.keys()
    ]
)
# Check each model version has a filename, and url or local path
for model_dict in MODEL_TASK_VERSIONS.values():
    for task_dict in model_dict.values():
        for version_dict in task_dict.values():
            assert "filename" in version_dict
            assert "url" in version_dict or "dir" in version_dict

TASK_MODELS = defaultdict(list)
for model, task_dict in MODEL_TASK_VERSIONS.items():
    for task, version_dict in task_dict.items():
        TASK_MODELS[task].append(model)

# Define namedtuple structure for a model parameter
Param = namedtuple("Param", ["arg", "default", "dtype", "tooltip"])
# Define each of the model parameters
sam_params = {
    "Points per side": Param("points_per_side", 32, int, ""),
    "Points per batch": Param("points_per_batch", 64, int, ""),
    "Pred IoU threshold": Param("pred_iou_thresh", 0.88, float, ""),
    "Stability score threshold": Param(
        "stability_score_thresh", 0.95, float, ""
    ),
    "Stability score offset": Param("stability_score_offset", 1.0, float, ""),
    "Box nms_thresh": Param("box_nms_thresh", 0.7, float, ""),
    "Crop N layers": Param("crop_n_layers", 0, int, ""),
    "Crop NMS thresh": Param("crop_nms_thresh", 0.7, float, ""),
    "Crop overlap ratio": Param(
        "crop_overlap_ratio", round(512 / 1500, 5), float, ""
    ),
    "Crop B points downscale factor": Param(
        "crop_n_points_downscale_factor", 1, int, ""
    ),
    "Min mask region area": Param("min_mask_region_area", 0, float, ""),
}
unet_params = {}
# Assuming every model version for each task has the same set of params
# TODO: Need a better solution long-term
MODEL_PARAMS = {
    "sam": {v: sam_params for v in MODEL_TASK_VERSIONS["sam"].keys()},
    "unet": {v: unet_params for v in MODEL_TASK_VERSIONS["unet"].keys()},
}

# Construct final mega-dict of model info
MODEL_INFO = {}

for model_name in MODEL_NAMES:
    MODEL_INFO[model_name] = {
        "display_name": MODEL_NAMES[model_name],
        "tasks": list(MODEL_TASK_VERSIONS[model_name].keys()),
        "versions": {
            task: list(task_dict.keys())
            for task, task_dict in MODEL_TASK_VERSIONS[model_name].items()
        },
        "params": MODEL_PARAMS[model_name],
    }
