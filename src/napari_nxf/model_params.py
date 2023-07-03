from qtpy.QtGui import QIntValidator, QDoubleValidator

VALIDATORS = {
    int: QIntValidator,
    float: QDoubleValidator
}

sam_params = [
    ("Points per side", "points_per_side", 32, int),
    ("Points per batch", "points_per_batch", 64, int),
    ("Pred IoU threshold", "pred_iou_thresh", 0.88, float),
    ("Stability score threshold", "stability_score_thresh", 0.95, float),
    ("Stability score offset", "stability_score_offset", 1.0, float),
    ("Box nms_thresh", "box_nms_thresh", 0.7, float),
    ("Crop N layers", "crop_n_layers", 0, int),
    ("Crop NMS thresh", "crop_nms_thresh", 0.7, float),
    ("Crop overlap ratio", "crop_overlap_ratio", round(512 / 1500, 5), float),
    ("Crop B points downscale factor", "crop_n_points_downscale_factor", 1, int),
    ("Min mask region area", "min_mask_region_area", 0, float),
]

unet_params = [
    ("Test 1", "test", 0.5, float),
    ("Test 2", "test2", 3, int),
]

MODEL_PARAMS = {
    "sam": sam_params,
    "unet": unet_params
}