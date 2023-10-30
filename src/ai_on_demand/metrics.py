import numpy as np


def dice(masks1: np.ndarray, masks2: np.ndarray):
    """Dice coefficient, a quotient of similarity in the range [0, 1]. 1 is perfect overlap, 0 is no overlap."""
    intersection = np.sum(np.logical_and(masks1, masks2))
    return 2 * intersection / (np.sum(masks1) + np.sum(masks2))


def iou(masks1: np.ndarray, masks2: np.ndarray):
    """Intersection over union, or Jaccard index, measures the overlap between two masks. Correlated with Dice, but slightly harsher on mistakes."""
    intersection = np.sum(np.logical_and(masks1, masks2))
    union = np.sum(np.logical_or(masks1, masks2))
    return intersection / union


def precision(preds: np.ndarray, labels: np.ndarray):
    """Precision, or positive predictive value, measures the proportion of predicted positives that are true positives."""
    tp = np.sum(np.logical_and(preds, labels))
    fp = np.sum(np.logical_and(preds == 1, labels == 0))
    return tp / (tp + fp)


def recall(preds: np.ndarray, labels: np.ndarray):
    """Recall, or sensitivity, measures the proportion of true positives that are predicted positives."""
    tp = np.sum(np.logical_and(preds, labels))
    fn = np.sum(np.logical_and(preds == 0, labels == 1))
    return tp / (tp + fn)


def labelled_to_binary(masks: np.ndarray):
    # Convert labelled instance masks to a flat binary mask
    return (masks > 0).astype(int)


if __name__ == "__main__":
    masks1_fpath = "/Users/shandc/Documents/ai_ondemand/example_files/example_stack_masks_mito-mitonet-MitoNet-v1.npy"
    masks2_fpath = "/Users/shandc/Documents/ai_ondemand/example_files/example_stack2_masks_mito-mitonet-MitoNet-v1.npy"

    masks1 = np.load(masks1_fpath)
    masks2 = np.load(masks2_fpath)

    # Convert to binary masks
    masks1_bin = labelled_to_binary(masks1)
    masks2_bin = labelled_to_binary(masks2)

    # Calculate metrics
    dice_score = dice(masks1_bin, masks2_bin)
    iou_score = iou(masks1_bin, masks2_bin)
    precision_score = precision(masks1_bin, masks2_bin)
    recall_score = recall(masks1_bin, masks2_bin)

    print(
        f"Dice: {dice_score}\nIoU: {iou_score}\nPrecision: {precision_score}\nRecall: {recall_score}"
    )
    breakpoint()
