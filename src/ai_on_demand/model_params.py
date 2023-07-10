from collections import namedtuple

Param = namedtuple("Param", ["arg", "default", "dtype", "tooltip"])

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

MODEL_VERSIONS = {
    "sam": ["default", "vit_l", "vit_b", "MedSAM"],
    "unet": ["UNet", "Attention UNet"],
}
