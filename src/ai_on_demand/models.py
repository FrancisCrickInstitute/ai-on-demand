from collections import namedtuple, defaultdict
from pathlib import Path

from ai_on_demand.tasks import TASK_NAMES
from ai_on_demand.utils import filter_empty_dict

# Define model shorthand and display names
MODEL_NAMES = {
    "sam": "Segment Anything",
    "unet": "U-Net",
    "mitonet": "MitoNet",
}

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
            "U-Net": {
                "filename": "mito_5nm_intensity_augs_warp.best.969.pt",
                "dir": "/nemo/stp/ddt/working/shandc/aiod_models/",
                "config": "mito_5nm_intensity_augs_warp.yml",
                # "dir": "/Volumes/ddt/working/shandc/aiod_models/",
            },
            "Attention U-Net": {
                "filename": "Attention_HUNet_3e5_Adam_restart_12_16.best.1266.pt",
                "dir": "/nemo/stp/ddt/working/shandc/aiod_models/",
                "config": "Attention_HUNet_3e5_Adam_restart_12_16.yml",
                # "dir": "/Volumes/ddt/working/shandc/aiod_models/",
            },
        },
        "ne": {
            "Attention U-Net": {
                "filename": "Attention_HUNet_NE.best.368.pt",
                "dir": "/nemo/stp/ddt/working/shandc/aiod_models/",
                "config": "Attention_HUNet_NE.yml",
                # "dir": "/Volumes/ddt/working/shandc/aiod_models/",
            }
        },
    },
    "mitonet": {
        "mito": {
            "MitoNet v1": {
                "filename": "mitonet_v1.pt",
                "url": "https://zenodo.org/record/6861565/files/MitoNet_v1.pth?download=1",
            },
            "MitoNet Mini v1": {
                "filename": "mitonet_mini_v1.pt",
                "url": "https://zenodo.org/record/6861565/files/MitoNet_v1_mini.pth?download=1",
            },
        }
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
# If a local path, check whether it is accessible
invalid_models = []
for model_name, model_dict in MODEL_TASK_VERSIONS.items():
    for task_name, task_dict in model_dict.items():
        for version_name, version_dict in task_dict.items():
            assert "filename" in version_dict
            assert "url" in version_dict or "dir" in version_dict
            # Permissions errors can occur, but can occur as
            # a RuntimeError, not OSError, so just catch all
            try:
                if "dir" in version_dict:
                    if not Path(version_dict["dir"]).exists():
                        invalid_models.append(
                            (model_name, task_name, version_name)
                        )
            except Exception as e:
                print(
                    f"Error found with {(model_name, task_name, version_name)}: {e}"
                )
                invalid_models.append((model_name, task_name, version_name))
# Filter out models that are not available so they do not show in the UI
if len(invalid_models) > 0:
    for model_name, task_name, version_name in invalid_models:
        del MODEL_TASK_VERSIONS[model_name][task_name][version_name]
# This can lead to nested empty dicts, so filter them out
MODEL_TASK_VERSIONS = filter_empty_dict(MODEL_TASK_VERSIONS)

TASK_MODELS = defaultdict(list)
for model, task_dict in MODEL_TASK_VERSIONS.items():
    for task, version_dict in task_dict.items():
        TASK_MODELS[task].append(model)

# Define namedtuple structure for a model parameter
Param = namedtuple("Param", ["arg", "values", "dtype", "tooltip"])
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
# Internal U-Net doesn't really take in params, so leave empty for now
unet_params = {}
# Empanada/MitoNet params
mitonet_params = {
    "Mode": Param(
        "mode",
        ["2D", "3D"],
        str,
        "Specifies whether to segment each slice independently or the stack as a whole",
    ),
    "Downsampling": Param(
        "downsampling",
        [1, 2, 4, 8, 16, 32, 64],
        int,
        "Downsampling factor for the input image",
    ),
    "Segmentation threshold": Param(
        "conf_threshold",
        0.5,
        float,
        "Confidence threshold for the segmentation",
    ),
    "Center threshold": Param(
        "center_threshold", 0.1, float, "Confidence threshold for the center"
    ),
    "Minimum distance": Param(
        "min_distance", 3, int, "Minimum distance between object centers"
    ),
    "Maximum objects": Param(
        "max_objects",
        1000,
        int,
        "Maximum number of objects to segment per class",
    ),
    "Semantic only": Param(
        "semantic_only",
        False,
        bool,
        "Only run semantic segmentation for all classes",
    ),
    "Fine boundaries": Param(
        "fine_boundaries", False, bool, "Finer boundaries between objects"
    ),
}
MODEL_PARAMS_DICT = {
    "sam": sam_params,
    "unet": unet_params,
    "mitonet": mitonet_params,
}

# Filter down to only the models that have at least one task available
VALID_MODEL_NAMES = {
    k: v for k, v in MODEL_NAMES.items() if k in MODEL_TASK_VERSIONS
}
# Reverse dict for convenience
MODEL_DISPLAYNAMES = {v: k for k, v in VALID_MODEL_NAMES.items()}

# Assuming every model version for each task has the same set of params
# TODO: Need a better solution long-term
MODEL_PARAMS = {}
for model_name, param_dict in MODEL_PARAMS_DICT.items():
    if model_name not in VALID_MODEL_NAMES:
        continue
    else:
        MODEL_PARAMS[model_name] = {
            v: param_dict for v in MODEL_TASK_VERSIONS[model_name].keys()
        }

# Construct final mega-dict of model info
MODEL_INFO = {}

for model_name in VALID_MODEL_NAMES:
    MODEL_INFO[model_name] = {
        "display_name": VALID_MODEL_NAMES[model_name],
        "tasks": list(MODEL_TASK_VERSIONS[model_name].keys()),
        "versions": {
            task: list(task_dict.keys())
            for task, task_dict in MODEL_TASK_VERSIONS[model_name].items()
        },
        "params": MODEL_PARAMS[model_name],
    }
