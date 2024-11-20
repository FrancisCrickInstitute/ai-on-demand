from pathlib import Path
import time
from typing import Optional, Union

import napari
from napari.qt.threading import thread_worker
import numpy as np

from ai_on_demand.tasks import TaskWidget
from ai_on_demand.data_selection import DataWidget
from ai_on_demand.mask_export import ExportWidget
from ai_on_demand.model_selection import ModelWidget
from ai_on_demand.nxf import NxfWidget
from ai_on_demand.preprocess import PreprocessWidget
from ai_on_demand.widget_classes import MainWidget
from ai_on_demand.utils import calc_param_hash
import aiod_utils.preprocess
from aiod_utils.io import extract_idxs_from_fname
import aiod_utils.rle as aiod_rle


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
        self.run_hash = None

        # Set selection colour
        self.colour_selected = "#F7AD6F"

        # Create radio buttons for selecting task (i.e. organelle)
        self.register_widget(
            TaskWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.register_widget(
            ModelWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Create the box for selecting the directory, showing img count etc.
        self.register_widget(
            DataWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Add a box for preprocessing options
        self.register_widget(
            PreprocessWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Add the button for running the Nextflow pipeline
        self.register_widget(
            NxfWidget(
                viewer=self.viewer,
                parent=self,
                pipeline="inference",
                expanded=False,
            )
        )

        # Add box for exporting masks
        self.register_widget(
            ExportWidget(viewer=self.viewer, parent=self, expanded=False)
        )

    def get_run_hash(self, nxf_params: dict):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        hashed_params = {}
        # Add model details
        hashed_params["task"] = nxf_params["task"]
        hashed_params["model"] = nxf_params["model"]
        hashed_params["variant"] = nxf_params["model_type"]
        # Add the model dictionary (hashed)
        hashed_params["model_hash"] = self.subwidgets["model"].model_param_hash
        # Get the advanced Nextflow parameters
        hashed_params.update(
            {
                k: v
                for k, v in nxf_params.items()
                if k in ["num_substacks", "overlap"]
            }
        )
        # Get the preprocessing parameters
        hashed_params["preprocess"] = nxf_params["preprocess"]
        # Though this only applies if post-processing is added (I think)
        if self.subwidgets["nxf"].postprocess_btn.isChecked():
            hashed_params["iou_threshold"] = nxf_params["iou_threshold"]
        # Calculate the overall hash for this run considering the model parameters
        # and Nextflow parameters that affect the output
        self.run_hash = calc_param_hash(hashed_params)

    def check_masks(self) -> tuple[bool, list, list]:
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
                mask_data = aiod_rle.load_encoding(mask_fpath)
                mask_data, metadata = aiod_rle.decode(mask_data)
                # Check if the mask layer already exists
                if layer_name in self.viewer.layers:
                    # If so, update the data just to make sure & ensure visible
                    self.viewer.layers[layer_name].data = mask_data
                    self.viewer.layers[layer_name].visible = True
                # If not, add a Labels layer
                else:
                    # Add a Labels layer for this file
                    self.viewer.add_labels(
                        mask_data,
                        name=layer_name,
                        visible=True,
                        opacity=0.5,
                        metadata=metadata["metadata"],
                    )
            else:
                # If the associated image is present, use its shape
                # Get ndim of the layer (this accounts for RGB)
                ndim = self.viewer.layers[f"{fpath.stem}"].ndim
                metadata = self.viewer.layers[f"{fpath.stem}"].metadata
                # Channels (non-RGB) & Z
                if ndim == 4:
                    # Channels should be first, don't care for labels so remove
                    img_shape = self.viewer.layers[f"{fpath.stem}"].data.shape[
                        1:
                    ]
                elif ndim == 3:
                    # If we have a Z, no problem
                    if ("bioio_dims" in metadata) and (
                        metadata["bioio_dims"].Z > 1
                    ):
                        img_shape = self.viewer.layers[
                            f"{fpath.stem}"
                        ].data.shape
                    # Otherwise it's a non-RGB multi-channel 2D image
                    else:
                        img_shape = self.viewer.layers[
                            f"{fpath.stem}"
                        ].data.shape[1:]
                # Otherwise take the 2D image shape
                # NOTE: [:ndim] is to handle RGB images as Napari interprets
                else:
                    img_shape = self.viewer.layers[f"{fpath.stem}"].data.shape[
                        :ndim
                    ]
                # Get the resulting shape considering preprocessing options
                options = self.subwidgets["preprocess"].extract_options()
                # Check if downsampling
                metadata = {}
                downsample_factor = (
                    aiod_utils.preprocess.get_downsample_factor(options)
                )
                if downsample_factor is not None:
                    metadata["downsample_factor"] = downsample_factor
                mask_shape = aiod_utils.preprocess.get_output_shape(
                    options=options, input_shape=img_shape
                )
                # Add a Labels layer for this file
                self.viewer.add_labels(
                    np.zeros(mask_shape, dtype=np.uint16),
                    name=layer_name,
                    visible=False,
                    opacity=0.5,
                    metadata=metadata,
                )
            # Move this layer to the top
            self.viewer.layers.move_multiple(
                [
                    self.viewer.layers.index(Path(fpath).stem),
                    self.viewer.layers.index(layer_name),
                ],
                -1,
            )

    def watch_mask_files(self):
        """
        File watcher to watch for new mask files being created during the Nextflow run.

        This is used to update the napari Labels layers with the new masks.

        Currently expects that the slices are stored as .rle files. Deactivates
        when it sees each image has the expected number of slices completed.
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
                    self.subwidgets["nxf"].mask_dir_path.glob("*.rle")
                )
                # Filter out any _all files, can occur when process is too fast (i.e. single image)
                current_files = [
                    i for i in current_files if Path(i).stem[-4:] != "_all"
                ]
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
                # If we have as many slices as the total, we are done
                if (
                    sum(self.subwidgets["nxf"].progress_dict.values())
                    == self.subwidgets["nxf"].total_substacks
                ):
                    print("Deactivating watcher...")
                    self.watcher_enabled = False

        # Call the nested function
        _watch_mask_files(self)

    def _get_mask_layer_name(
        self,
        stem: str,
        extension: Optional[str] = None,
        executed: bool = False,
        include_hash: bool = True,
        truncate: bool = True,
    ):
        # If executed, use the executed attributes in case the user has changed the selection since running the pipeline
        task_model_variant_name = self.subwidgets[
            "model"
        ].get_task_model_variant_name(executed)
        # Construct the mask layer name
        fname = f"{stem}_masks_{task_model_variant_name}"
        # Add the hash if requested
        if include_hash:
            if truncate:
                fname += f"-{self.run_hash[:8]}"
            else:
                fname += f"-{self.run_hash}"
        if extension is not None:
            fname += f".{extension}"
        return fname

    def _get_mask_name(
        self, stem: str, extension: str = "rle", executed=False, truncate=False
    ):
        mask_root = self._get_mask_layer_name(
            stem=stem, executed=executed, truncate=truncate
        )
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

    def update_masks(self, new_files: list[Union[str, Path]]):
        """
        Update the masks in the napari Labels layers with the new masks found in the last scan.
        """
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            try:
                mask_arr = aiod_rle.load_encoding(f)
                mask_arr, _ = aiod_rle.decode(mask_arr)
            # NOTE: This is a temporary fix, and only occurs with fast models and a good GPU
            except FileNotFoundError:
                print(
                    f"File {f} not found, may have already been deleted. Skipping..."
                )
                continue
            except ValueError as e:
                print(f)
                print(e)
                continue
            # Need to get the preprocessing options to check if downsampling was used
            preprocess_params = aiod_utils.preprocess.load_methods(
                self.subwidgets["preprocess"].extract_options()
            )
            # Get the downsample factor
            downsample_factor = aiod_utils.preprocess.get_downsample_factor(
                preprocess_params
            )
            # Get indices from fname, modified if downsampled
            start_x, end_x, start_y, end_y, start_z, end_z = (
                extract_idxs_from_fname(
                    fname=f, downsample_factor=downsample_factor
                )
            )
            # Check if the mask layer has been renamed
            prefix, suffix = f.stem.split("_masks_")
            # Extract the relevant Labels layer
            mask_layer_name = self._get_mask_layer_name(prefix, executed=True)
            label_layer = self.viewer.layers[mask_layer_name]
            # Insert mask data
            # Check if dims match
            if label_layer.ndim != mask_arr.ndim:
                mask_arr = np.squeeze(mask_arr)
                assert (
                    label_layer.ndim == mask_arr.ndim
                ), f"Mask appears to be {mask_arr.ndim}D (after squeezing), but layer is {label_layer.ndim}D"
                label_layer.data = mask_arr
            else:
                # TODO: Handle multi-channel images
                # TODO: Check DHW orientation? Does Napari enforce this?
                if label_layer.ndim == 3:
                    label_layer.data[
                        start_z:end_z, start_x:end_x, start_y:end_y
                    ] = mask_arr
                else:
                    label_layer.data[start_x:end_x, start_y:end_y] = mask_arr
            label_layer.visible = True
            # Try to rearrange the layers to get them on top
            idxs = []
            # Have to check due to possible delay in loading
            if prefix in self.viewer.layers:
                img_idx = self.viewer.layers.index(self.viewer.layers[prefix])
                idxs.append(img_idx)
            # We create the mask layer, so it will always exist
            label_idx = self.viewer.layers.index(label_layer)
            idxs.append(label_idx)
            self.viewer.layers.move_multiple(idxs, -1)
            # Switch viewer to latest slice
            self.viewer.dims.set_point(0, end_z - 1)
            # Insert the slice number into tracker for the progress bar
            self.subwidgets["nxf"].progress_dict[prefix] += 1
        # Now update the total progress bar
        self.subwidgets["nxf"].update_progress_bar()

    def insert_final_masks(self):
        """
        Insert the final masks into the napari Labels layers.

        This is used to update the napari Labels layers with the final masks
        after the Nextflow pipeline has completed.
        """
        # Loop over each image and insert the final mask
        for img_name, img_fpath in self.subwidgets[
            "data"
        ].image_path_dict.items():
            # Get the mask layer name
            mask_layer_name = self._get_mask_layer_name(
                img_name, executed=True
            )
            # Clear the current mask layer of data (to free up memory??)
            self.viewer.layers[mask_layer_name].data = np.zeros_like(
                self.viewer.layers[mask_layer_name].data
            )
            # Load the mask
            fpath = self.subwidgets["nxf"].mask_dir_path / self._get_mask_name(
                img_fpath.stem, executed=True, truncate=False
            )
            mask_arr = aiod_rle.load_encoding(fpath)
            # NOTE: Mask metadata should be no different, so ignore
            mask_arr, _ = aiod_rle.decode(mask_arr)
            # Insert mask data
            self.viewer.layers[mask_layer_name].data = mask_arr
            self.viewer.layers[mask_layer_name].visible = True
