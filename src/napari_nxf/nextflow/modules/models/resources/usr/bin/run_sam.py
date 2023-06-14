#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage.io

from napari.layers.image._image_utils import guess_rgb
from napari.layers.utils.layer_utils import calc_data_range

SAM_MODELS = {
    "default": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "MedSAM": {
        "filename": "sam_vit_b_01ec64_medsam.pth",
        "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth"
    },
}

def run_sam(root_dir, fpath, model_type):
    sam = sam_model_registry[model_type](
        checkpoint=root_dir / "sam_chkpts" / SAM_MODELS[model_type]["filename"]
    )
    # Create the model
    model = SamAutomaticMaskGenerator(sam)
    # Load the image
    img = skimage.io.imread(fpath)
    # Extract the dimensions
    ndim = img.ndim
    # Reduce ndims if RGB (i.e. it's a single RGB image, not a stack)
    if guess_rgb(img.shape):
        ndim -= 1
    # Send the image to the corresponding run func based on slice or stack
    if ndim == 2:
        all_masks = _run_sam_slice(img, model)
    elif ndim == 3:
        all_masks = _run_sam_stack(root_dir, img, model, fpath)
    else:
        raise ValueError("Can only handle an image, or stack of images!")
    save_masks(root_dir, fpath, all_masks, all=True)
    return img, all_masks

def _run_sam_slice(img_slice, model):
    # Expand to 3-channel if not rgb
    if not guess_rgb(img_slice.shape):
        img_slice = np.stack((img_slice,) * 3, axis=-1)
    img_slice = img_slice[..., :3]
    masks = model.generate(img_slice)
    # Convert the masks into a napari-friendly format
    mask_img = create_mask_arr(masks)
    return mask_img

def _run_sam_stack(root_dir, img_stack, model, fpath):
    # Initialize the container of all masks
    all_masks = np.zeros(img_stack.shape, dtype=int)
    # Use napari function to extract the contrast limits
    # Avoids any mismatches further down the line
    contrast_limits = calc_data_range(img_stack)
    # Loop over each stack and run
    for idx in range(img_stack.shape[0]):
        img_slice = img_stack[idx]
        # Expand to 3-channel if not rgb
        if not guess_rgb(img_slice.shape):
            img_slice = np.stack((img_slice,) * 3, axis=-1)
        img_slice = img_slice[..., :3]
        # Normalize the slice
        # Convert to uint8 just in case
        img_slice = normalize_slice(
            img_slice,
            source_limits=contrast_limits,
            target_limits=(0,255)
        ).astype(np.uint8)
        # Actually run the model on this slice
        masks = _run_sam_slice(img_slice, model)
        # Insert the masks for this slice
        all_masks[idx, ...] = masks
        # Don't save final slice
        print(idx, img_stack.shape[0])
        if idx < img_stack.shape[0]-1:
            save_masks(
                root_dir=root_dir,
                fpath=fpath,
                masks=all_masks,
                stack_slice=True,
                idx=idx
            )
        else:
            # Remove the penultimate slice
            (root_dir / "sam_masks" / f"{Path(fpath).stem}_masks_{idx-1}.npy").unlink()
    # Align masks to ensure consistent colouration
    all_masks = align_segment_labels(all_masks)
    return all_masks

# def align_segment_labels(all_masks):
#     aligned_masks = all_masks.copy()

#     for idx in range(all_masks.shape[0] - 1):
#         curr_slice = all_masks[idx]
#         next_slice = all_masks[idx+1]
#         curr_labels, curr_counts = np.unique(curr_slice, return_counts=True)
#         next_labels, next_counts = np.unique(next_slice, return_counts=True)

#         cost_matrix = np.zeros((
#             curr_labels.size,
#             next_labels.size
#         ), dtype=int)

#         for label in next_labels:
#             cost_matrix[curr_slice == label] += 1

def align_segment_labels(all_masks, threshold=0.5):
    # From https://github.com/MIC-DKFZ/napari-sam/blob/main/src/napari_sam/_widget.py#L1118
    '''
    There is a potentially better way to do this, using the Hungarian algorithm
    It will, however, still require computing the "cost" (i.e. overlap, defined as
    the count of co-occurences between every numerical label between two slices)
    The Hungarian algorithm itself can be easily done using scipy.optimize.linear_sum_assignment
    It's just that then the optimal assignment will be found, rather than using this
    thresholded approach. Can revise later as needed.
    '''
    all_masks = all_masks.copy()
    for i in range(all_masks.shape[0] - 1):
        current_slice = all_masks[i]
        next_slice = all_masks[i+1]
        next_labels, next_label_counts = np.unique(next_slice, return_counts=True)
        next_label_counts = next_label_counts[next_labels != 0]
        next_labels = next_labels[next_labels != 0]
        new_next_slice = np.zeros_like(next_slice)
        if len(next_labels) > 0:
            for next_label, next_label_count in zip(next_labels, next_label_counts):
                current_roi_labels = current_slice[next_slice == next_label]
                current_roi_labels, current_roi_label_counts = np.unique(current_roi_labels, return_counts=True)
                current_roi_label_counts = current_roi_label_counts[current_roi_labels != 0]
                current_roi_labels = current_roi_labels[current_roi_labels != 0]
                if len(current_roi_labels) > 0:
                    current_max_count = np.max(current_roi_label_counts)
                    current_max_count_label = current_roi_labels[np.argmax(current_roi_label_counts)]
                    overlap = current_max_count / next_label_count
                    if overlap >= threshold:
                        new_next_slice[next_slice == next_label] = current_max_count_label
                    else:
                        new_next_slice[next_slice == next_label] = next_label
                else:
                    new_next_slice[next_slice == next_label] = next_label
            all_masks[i+1] = new_next_slice
    return all_masks

def normalize_slice(img_slice, source_limits, target_limits):
    # From https://github.com/MIC-DKFZ/napari-sam/blob/main/src/napari_sam/utils.py
    if source_limits is None:
        source_limits = (img_slice.min(), img_slice.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return img_slice * 0
    else:
        x_std = (img_slice - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled

def create_mask_arr(masks):
    # Sort the masks/annotations by area to allow overwriting/lapping
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    # Need integers for napari Labels layer
    # NOTE: argmax used in other libraries, but wrong due to double-use of 0
    # That will result in 1 less mask present
    mask_img = np.zeros(sorted_anns[0]["segmentation"].shape, dtype=int)
    for i, mask in enumerate(sorted_anns):
        mask_img[mask["segmentation"]] = i+1
    return mask_img

def save_masks(root_dir, fpath, masks, stack_slice=False, all=False, idx=None):
    save_dir = root_dir / "sam_masks"
    save_dir.mkdir(parents=True, exist_ok=True)
    # Cannot save a slice of a stack and all slice(s)
    assert not(stack_slice and all)
    # Incrementally save the masks of a slice from a stack
    if stack_slice:
        save_path = save_dir / f"{Path(fpath).stem}_masks_{idx}.npy"
        # Remove file for previous mask iteration
        if idx > 0:
            (save_dir / f"{Path(fpath).stem}_masks_{idx-1}.npy").unlink()
    # Specify path for img or all slices, indicating finished
    if all:
        save_path = save_dir / f"{Path(fpath).stem}_masks_all.npy"
    # Save the masks!
    np.save(save_path, masks)

def plot_masks(img, masks):
    fig, ax = plt.subplots(1, 1, figsize=(20,20))
    # Sort the masks/annotations by largest area to allow overwriting/lapping
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    # Create the image array for the segmentations themselves to overlay
    mask_img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # Set transparency
    mask_img[:, :, 3] = 0
    # Loop over each segmentation and insert a random colour at that location
    for ann in sorted_anns:
        mask_img[ann["segmentation"]] = np.concatenate([
            np.random.random(3),  # Random RGB colour
            [0.35]  # Transparency/alpha
        ])
    ax.imshow(img)
    ax.set_axis_off()
    ax.imshow(mask_img)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--model", help="Select model type", default="default")
    parser.add_argument("--config", help="Model parameter config path")

    cli_args = parser.parse_args()

    # Kill me
    root_dir = Path(__file__).parent.parent.parent.parent
    print(root_dir)

    img, masks = run_sam(
        root_dir=root_dir,
        fpath=cli_args.path,
        model_type=cli_args.model
    )
