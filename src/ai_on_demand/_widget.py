from itertools import compress
from pathlib import Path
import time
from typing import Optional

import napari
from napari.qt.threading import thread_worker
import numpy as np

from ai_on_demand.tasks import TaskWidget
from ai_on_demand.data_selection import DataWidget
from ai_on_demand.model_selection import ModelWidget
from ai_on_demand.nxf import NxfWidget
from ai_on_demand.utils import sanitise_name
from ai_on_demand.widget_classes import MainWidget


class Inference(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer=napari_viewer,
            title="Inference",
            tooltip="""
Run segmentation/inference on selected images using one of the available pre-trained models.
""",
        )
        # Handy attributes to check things
        # These get set in subwidgets, but are used across so set here for ease
        self.selected_task = None
        self.selected_model = None
        self.selected_variant = None
        self.executed_task = None
        self.executed_model = None
        self.executed_variant = None

        # Set selection colour
        self.colour_selected = "#F7AD6F"

        # Create radio buttons for selecting task (i.e. organelle)
        self.register_widget(TaskWidget(viewer=self.viewer, parent=self))

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.register_widget(ModelWidget(viewer=self.viewer, parent=self))

        # Create the box for selecting the directory, showing img count etc.
        self.register_widget(DataWidget(viewer=self.viewer, parent=self))

        # Add the button for running the Nextflow pipeline
        self.register_widget(
            NxfWidget(viewer=self.viewer, parent=self, pipeline="inference")
        )

    def check_masks(self) -> bool:
        """
        Function to check if masks are present for the current setup, either
        already imported or in the Nextflow output directory.

        If all are present, avoids running the Nextflow pipeline.
        """
        # List of booleans for whether masks exist for each image
        masks_exist = []
        # List of image paths to load masks for
        load_paths = []
        # List of image paths to pass to Nextflow
        img_paths = []
        for img_name, img_fpath in self.subwidgets[
            "data"
        ].image_path_dict.items():
            mask_layer_name = self._get_mask_layer_name(
                img_name, executed=True
            )
            # Check if this mask has been imported already
            if mask_layer_name in self.viewer.layers:
                masks_exist.append(True)
            # Check if the mask exists from a previous run to load in
            elif (
                self.subwidgets["nxf"].mask_dir_path
                / self._get_mask_name(img_name, executed=True)
            ).exists():
                masks_exist.append(True)
                load_paths.append(img_fpath)
            # Otherwise, we need to run the pipeline
            else:
                masks_exist.append(False)
                img_paths.append(img_fpath)
        # Proceed to run the pipeline if any masks are missing
        proceed = not all(masks_exist)
        # If we aren't proceeding, there should be no images without masks!
        if not proceed:
            assert len(img_paths) == 0
        return proceed, img_paths, load_paths

    def create_mask_layers(self, img_paths=None):
        if img_paths is None:
            img_paths = self.subwidgets["data"].image_path_dict.values()
        # Construct the mask layer names
        layer_names = [
            self._get_mask_layer_name(Path(i).stem, executed=True)
            for i in img_paths
        ]
        # Create the Labels layers for each image
        for fpath, layer_name in zip(img_paths, layer_names):
            # Check if the mask file already exists
            mask_fpath = self.subwidgets[
                "nxf"
            ].mask_dir_path / self._get_mask_name(fpath.stem, executed=True)
            # If it does, load it
            if mask_fpath.exists():
                mask_data = np.load(mask_fpath)
                # Check if the mask layer already exists
                if layer_name in self.viewer.layers:
                    # If so, update the data just to make sure & ensure visible
                    self.viewer.layers[layer_name].data = mask_data
                    self.viewer.layers[layer_name].visible = True
                # If not, add a Labels layer
                else:
                    # Add a Labels layer for this file
                    self.viewer.add_labels(
                        mask_data, name=layer_name, visible=True
                    )
            else:
                # If the associated image is present, use its shape
                try:
                    img_shape = self.viewer.layers[f"{fpath.name}"].data.shape
                # Otherwise default to 1000x1000 to avoid weird viewer
                except KeyError:
                    img_shape = (1000, 1000)
                # Add a Labels layer for this file
                self.viewer.add_labels(
                    np.zeros(img_shape, dtype=int),
                    name=layer_name,
                    visible=False,
                )
            # Move this layer to the top
            self.viewer.layers.move_multiple(
                [
                    self.viewer.layers.index(Path(fpath).stem),
                    self.viewer.layers.index(layer_name),
                ]
            )

    def watch_mask_files(self):
        """
        File watcher to watch for new mask files being created during the Nextflow run.

        This is used to update the napari Labels layers with the new masks.

        Currently expects that the slices are stored as .npy files. Deactivates
        when it sees each image has an associated "*_all.npy" file.
        """
        # Wait for at least one image to load as layers if not present
        if not self.viewer.layers:
            time.sleep(1)
        # Create the Labels layers for each image
        self.create_mask_layers()

        # NOTE: Wrapper as self/class not available at runtime
        @thread_worker(
            connect={
                "yielded": self.update_masks,
                "returned": self._reset_viewer,
            }
        )
        def _watch_mask_files(self):
            # Enable the watcher
            print("Activating watcher...")
            self.watcher_enabled = True
            # Initialize empty container for storing mask filepaths
            self.mask_fpaths = []
            # Loop and yield any changes infinitely while enabled
            while self.watcher_enabled:
                # Get all files
                current_files = list(
                    self.subwidgets["nxf"].mask_dir_path.glob("*.npy")
                )
                # Filter out files we are not running on
                current_files = [
                    i
                    for i in current_files
                    if Path(i).stem.split("_masks_")[0]
                    in self.subwidgets["data"].image_path_dict
                ]
                if set(self.mask_fpaths) != set(current_files):
                    # Get the new files only
                    new_files = [
                        i for i in current_files if i not in self.mask_fpaths
                    ]
                    # Update file list and yield the difference
                    self.mask_fpaths = current_files
                    if new_files:
                        yield new_files
                # Sleep until next check
                time.sleep(2)
                # Check all masks contain data for all slices
                masks_finished = [
                    Path(i).stem[-3:] == "all" for i in current_files
                ]
                # Get how many complete mask files there should be
                num_images = len(self.subwidgets["data"].image_path_dict)
                # If all images have complete masks, deactivate watcher
                if all(masks_finished) and (len(masks_finished) == num_images):
                    print("Deactivating watcher...")
                    self.watcher_enabled = False

        # Call the nested function
        _watch_mask_files(self)

    def _get_mask_layer_name(
        self,
        stem: str,
        extension: Optional[str] = None,
        executed: bool = False,
    ):
        # If executed, use the executed attributes in case the user has changed the selection since running the pipeline
        if executed:
            fname = f"{stem}_masks_{self.executed_task}-{self.executed_model}-{sanitise_name(self.executed_variant)}"
        else:
            fname = f"{stem}_masks_{self.selected_task}-{self.selected_model}-{sanitise_name(self.selected_variant)}"
        if extension is not None:
            fname += f".{extension}"
        return fname

    def _get_mask_name(self, stem: str, extension="npy", executed=False):
        mask_root = self._get_mask_layer_name(stem=stem, executed=executed)
        # Add the _all marker to signify all slices/completeness
        mask_root += "_all"
        # Add the extension
        return f"{mask_root}.{extension}"

    def _reset_viewer(self):
        """
        Should help alleviate rendering issue where masks are mis-aligned.

        Need to do it here as interacting with the viewer in the thread_worker causes issues.
        """
        self.viewer.dims.set_point(0, 0)

    def update_masks(self, new_files):
        """
        Update the masks in the napari Labels layers with the new masks found in the last scan.
        """
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            try:
                mask_arr = np.load(f)
            # NOTE: This is a temporary fix, and only occurs with fast models and a good GPU
            except FileNotFoundError:
                print(
                    f"File {f} not found, may have already been deleted. Skipping..."
                )
                continue
            # Check if the mask layer has been renamed
            prefix = f.stem.split("_masks_")[0]
            # Extract the relevant Labels layer
            mask_layer_name = self._get_mask_layer_name(prefix, executed=True)
            label_layer = self.viewer.layers[mask_layer_name]
            # Insert mask data
            label_layer.data = mask_arr
            label_layer.visible = True
            # Try to rearrange the layers to get them on top
            idxs = []
            # Have to check due to possible delay in loading
            if prefix in self.viewer.layers:
                img_idx = self.viewer.layers.index(self.viewer.layers[prefix])
                idxs.append(img_idx)
            label_idx = self.viewer.layers.index(label_layer)
            idxs.append(label_idx)
            self.viewer.layers.move_multiple(idxs)
            slice_num = f.stem.split("_")[-1]
            # Switch viewer to latest slice
            if slice_num == "all":
                slice_num = label_layer.data.shape[0] - 1
            else:
                slice_num = int(slice_num)
            self.viewer.dims.set_point(0, slice_num)
            # Insert the slice number into tracker for the progress bar
            if prefix in self.subwidgets["nxf"].progress_dict:
                self.subwidgets["nxf"].progress_dict[prefix] = slice_num + 1
        # Now update the total progress bar
        self.subwidgets["nxf"].update_progress_bar()
