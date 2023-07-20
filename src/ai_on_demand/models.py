from collections import namedtuple, defaultdict

from .tasks import TASK_NAMES

# Define model shorthand and display names
MODEL_NAMES = {
    "sam": "Segment Anything",
    "unet": "U-Net",
}
# Reverse dict for convenience
MODEL_DISPLAYNAMES = {v: k for k, v in MODEL_NAMES.items()}

# Checker function so that all models have definitons below
def check_model_names(d):
    return all([k in MODEL_NAMES for k in d.keys()])


# Define the available versions of each model
MODEL_VERSIONS = {
    "sam": ["default", "vit_l", "vit_b", "MedSAM"],
    "unet": ["U-Net", "Attention U-Net"],
}
assert check_model_names(MODEL_VERSIONS)


# Checker function so that all model tasks are valid
def check_model_tasks(d):
    return all([task in TASK_NAMES for tasks in d.values() for task in tasks])


# Define the tasks each model can perform (at the meta-level)
# Each model may have variants that can only do one task, which is fine, but
# full capability is listed here
MODEL_TASKS = {
    "sam": ["everything"],
    "unet": ["mito", "er", "ne"],
}
check_model_tasks(MODEL_TASKS)
# Create the reverse dict to check available models per task
TASK_MODELS = defaultdict(list)
for model, tasks in MODEL_TASKS.items():
    for task in tasks:
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
unet_params = {
    "Test 1": Param("test", 0.5, float, ""),
    "Test 2": Param("test2", 3, int, ""),
}

MODEL_PARAMS = {"sam": sam_params, "unet": unet_params}
assert check_model_names(MODEL_VERSIONS)

# Construct final mega-dict of model info
MODEL_INFO = {}

for model_name in MODEL_NAMES:
    MODEL_INFO[model_name] = {
        "display_name": MODEL_NAMES[model_name],
        "tasks": MODEL_TASKS[model_name],
        "versions": MODEL_VERSIONS[model_name],
        "params": MODEL_PARAMS[model_name],
    }
