from collections import defaultdict
from pathlib import Path
import subprocess
from typing import Optional, Union
from urllib.parse import urlparse

from aiod_registry import TASK_NAMES
import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
import numpy as np
import pandas as pd
import qtpy.QtCore
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
)
import skimage.io
import tqdm
import yaml

from ai_on_demand.utils import sanitise_name, format_tooltip, get_img_dims
from ai_on_demand.widget_classes import SubWidget

# We need to import from the submodule
# But it's not a package...lots of issues no __init__'ing can fix it seems
# And I don't want to touch sys.path
import importlib
import sys

# TODO: Create aiod_utils package to handle this
spec = importlib.util.spec_from_file_location(
    name="create_splits",
    location=Path(__file__).parent
    / "Segment-Flow/modules/models/resources/usr/bin/create_splits.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules["create_splits"] = module
spec.loader.exec_module(module)
# This will now be imported
from create_splits import generate_stack_indices, calc_num_stacks, Stack


class NxfWidget(SubWidget):
    _name = "nxf"

    def __init__(
        self,
        viewer: napari.Viewer,
        pipeline: str,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        # Define attributes that may be useful outside of this class
        # or throughout it
        self.nxf_repo = "FrancisCrickInstitute/Segment-Flow"
        # Set the base Nextflow command
        self.setup_nxf_dir_cmd()
        super().__init__(
            viewer=viewer,
            title="Run Pipeline",
            parent=parent,
            layout=layout,
            tooltip="""
Allows for the computational pipeline to be triggered, with different additional options depending on the main widget selected.
The profile determines where the pipeline is run.
""",
            **kwargs,
        )
        # Whether all images have been loaded
        # Needed to properly extract metadata
        self.all_loaded = False
        # Dictionary to monitor progress of each image
        self.progress_dict = {}

        self.pipeline = pipeline
        # Available pipelines and their funcs
        self.pipelines = {
            "inference": {
                "check": self.check_inference,
                "setup": self.setup_inference,
            },
            "finetuning": {
                "check": None,
                "setup": self.setup_finetuning,
            },
        }
        # Connect viewer to callbacks on events
        self.viewer.layers.selection.events.changed.connect(
            self.on_select_change
        )

    def load_settings(self):
        """
        Load the settings for the plugin from the parent widget.
        """
        if not self.parent.plugin_settings:
            return
        if "nxf" in self.parent.plugin_settings:
            settings = self.parent.plugin_settings["nxf"]
            # Set the profile
            if "profile" in settings:
                idx = self.nxf_profile_box.findText(settings["profile"])
                if idx != -1:
                    self.nxf_profile_box.setCurrentIndex(idx)
            # Set the base directory
            if "base_dir" in settings:
                nxf_base_dir = Path(settings["base_dir"])
                self.nxf_dir_text.setText(str(nxf_base_dir))
                # Update the base directory and Nextflow command
                self.setup_nxf_dir_cmd(base_dir=Path(nxf_base_dir))

    def get_settings(self) -> dict:
        """
        Get the settings for the plugin to store for future sessions.
        """
        settings = {
            "base_dir": str(self.nxf_base_dir),
        }
        return settings

    def on_select_change(self, event):
        layers_selected = event.source
        # If nothing selected, reset the mask layers
        if len(layers_selected) == 0:
            # Filter mask layers to ensure they are from AIoD outputs and not external
            self.selected_mask_layers = self.parent.subwidgets[
                "data"
            ].get_mask_layers()
            # Reset text on export button
            self.export_masks_btn.setText("Export all masks")
            # Reset tile size label if nothing selected
            self.update_tile_size(val=None, clear_label=True)
        else:
            # Update the tile size label based on the selected layers
            self.update_tile_size(val=None, clear_label=False)
            # Filter mask layers to ensure they are from AIoD outputs and not external
            self.selected_mask_layers = self.parent.subwidgets[
                "data"
            ].get_mask_layers(layer_list=layers_selected)
            num_selected = len(self.selected_mask_layers)
            # In case non-Labels layers are selected, reset
            if num_selected == 0:
                self.selected_mask_layers = self.parent.subwidgets[
                    "data"
                ].get_mask_layers()
                self.export_masks_btn.setText("Export all masks")
            else:
                self.export_masks_btn.setText(
                    f"Export {num_selected} mask{'s' if num_selected > 1 else ''}"
                )
        return

    def setup_nxf_dir_cmd(self, base_dir: Optional[Path] = None):
        # Set the basepath to store masks/checkpoints etc. in
        if base_dir is not None:
            self.nxf_base_dir = base_dir
        else:
            self.nxf_base_dir = Path.home() / ".nextflow" / "aiod"
        self.nxf_base_dir.mkdir(parents=True, exist_ok=True)
        self.nxf_store_dir = self.nxf_base_dir / "aiod_cache"
        self.nxf_store_dir.mkdir(parents=True, exist_ok=True)
        # Set the base Nextflow command
        # Ensures logs are stored in the right place (must be before run)
        self.nxf_base_cmd = (
            f"nextflow -log '{str(self.nxf_base_dir / 'nextflow.log')}' "
        )
        # Path to store the text file containing the image paths
        self.img_list_fpath = self.nxf_store_dir / "all_img_paths.csv"
        # Working directory for Nextflow
        self.nxf_work_dir = self.nxf_base_dir / "work"
        self.nxf_work_dir.mkdir(parents=True, exist_ok=True)

    def create_box(self, variant: Optional[str] = None):
        # Create a drop-down box to select the execution profile
        self.nxf_profile_label = QLabel("Execution profile:")
        self.nxf_profile_label.setToolTip(
            format_tooltip("Select the execution profile to use.")
        )
        self.nxf_profile_box = QComboBox()
        # Get the available profiles from config dir
        config_dir = Path(__file__).parent / "Segment-Flow" / "profiles"
        avail_confs = [str(i.stem) for i in config_dir.glob("*.conf")]
        avail_confs.sort()
        self.nxf_profile_box.addItems(avail_confs)
        self.inner_layout.addWidget(self.nxf_profile_label, 0, 0)
        self.inner_layout.addWidget(self.nxf_profile_box, 0, 1)

        # Create the option for selecting base directory
        base_dir_layout = QGridLayout()
        self.nxf_dir_label = QLabel("Base directory:")
        base_dir_tooltip = "Select the base directory to store the Nextflow cache (i.e. all models & results) in."
        self.nxf_dir_label.setToolTip(format_tooltip(base_dir_tooltip))
        self.nxf_dir_text = QLabel(str(self.nxf_base_dir))
        self.nxf_dir_text.setWordWrap(True)
        self.nxf_dir_text.setToolTip(
            format_tooltip("The selected base directory.")
        )
        self.nxf_dir_text.setMaximumWidth(400)
        self.nxf_dir_btn = QPushButton("Change")
        self.nxf_dir_btn.clicked.connect(self.on_click_base_dir)
        self.nxf_dir_btn.setToolTip(format_tooltip(base_dir_tooltip))

        base_dir_layout.addWidget(self.nxf_dir_label, 0, 0, 1, 2)
        base_dir_layout.addWidget(self.nxf_dir_text, 0, 2, 1, 4)
        base_dir_layout.addWidget(self.nxf_dir_btn, 0, 6, 1, 1)
        self.inner_layout.addLayout(base_dir_layout, 1, 0, 1, 2)

        # Add a checkbox for overwriting existing results
        self.overwrite_btn = QCheckBox("Overwrite existing results")
        self.overwrite_btn.setToolTip(
            format_tooltip(
                """
Select/enable to overwrite any previous results.

Exactly what is overwritten will depend on the pipeline selected. By default, any previous results matching the current setup will be loaded if possible. This can be disabled by ticking this box.
        """
            )
        )
        self.inner_layout.addWidget(self.overwrite_btn, 2, 0, 1, 1)
        # Add a button for importing masks
        self.import_masks_btn = QPushButton("Import masks")
        self.import_masks_btn.clicked.connect(self.on_click_import)
        self.import_masks_btn.setToolTip(
            format_tooltip("Import segmentation masks.")
        )
        self.import_masks_btn.setEnabled(True)
        self.inner_layout.addWidget(self.import_masks_btn, 2, 1, 1, 1)

        # Add widget for advanced options
        self.options_widget = QWidget()
        self.options_layout = QVBoxLayout()
        self.advanced_box = QPushButton(" ▶ Advanced Options")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setStyleSheet(
            f"QPushButton {{ text-align: left; }} QPushButton:checked {{background-color: {self.parent.subwidgets['model'].colour_selected}}}"
        )
        self.advanced_box.toggled.connect(self.on_toggle_advanced)
        self.advanced_box.setToolTip(
            format_tooltip(
                """
Show/hide advanced options for the Nextflow pipeline. These options define how to split an image into separate jobs in Nextflow. The underlying models will likely do their own splitting internally into patches, but this controls the trade-off between the number and size of each job.
"""
            )
        )
        self.advanced_widget = QWidget()
        self.advanced_layout = QGridLayout()

        # Add the advanced options
        # Moved out due to length
        self._add_advanced_options()

        self.advanced_widget.setLayout(self.advanced_layout)
        self.advanced_widget.setVisible(False)
        self.options_layout.addWidget(self.advanced_box)
        self.options_layout.addWidget(self.advanced_widget)
        self.options_layout.setContentsMargins(0, 0, 0, 0)
        self.options_widget.setLayout(self.options_layout)
        self.inner_layout.addWidget(self.options_widget, 3, 0, 1, 2)

        # Create a button to navigate to a directory to take images from
        self.nxf_run_btn = QPushButton("Run Pipeline!")
        self.nxf_run_btn.clicked.connect(self.run_pipeline)
        self.nxf_run_btn.setToolTip(
            format_tooltip(
                "Run the pipeline with the chosen organelle(s), model, and images."
            )
        )
        self.inner_layout.addWidget(self.nxf_run_btn, 4, 0, 1, 2)

        # Add a button for exporting masks, with a dropdown for different formats
        # and checkbox for binarising
        export_layout = QHBoxLayout()
        self.export_masks_btn = QPushButton("Export all masks")
        self.export_masks_btn.clicked.connect(self.on_click_export)
        self.export_masks_btn.setToolTip(
            format_tooltip(
                "Export the output segmentation masks to a directory. Exports all masks by default, or only the selected masks (if any)."
            )
        )
        self.export_masks_btn.setEnabled(False)
        export_layout.addWidget(self.export_masks_btn)

        self.export_format_dropdown = QComboBox()
        self.export_format_dropdown.addItems([".npy", ".tiff"])
        export_layout.addWidget(self.export_format_dropdown)

        self.export_binary_check = QCheckBox("Binarise masks?")
        self.export_binary_check.setToolTip(
            format_tooltip(
                "Binarise the masks before exporting (i.e. black background, white masks)."
            )
        )
        export_layout.addWidget(self.export_binary_check)
        self.inner_layout.addLayout(export_layout, 5, 0, 1, 2)

        pbar_layout = QHBoxLayout()
        # Add progress bar
        self.pbar = QProgressBar()
        # Create the label associated with the progress bar
        self.pbar_label = QLabel("Progress: [--:--]")
        self.pbar_label.setToolTip(
            format_tooltip("Shows [elapsed<remaining] time for current run.")
        )
        # Add the label and progress bar to the layout
        pbar_layout.addWidget(self.pbar_label)
        pbar_layout.addWidget(self.pbar)
        self.inner_layout.addLayout(pbar_layout, 6, 0, 1, 2)
        # TQDM progress bar to monitor completion time
        self.tqdm_pbar = None
        # Add the layout to the main layout
        self.inner_widget.setLayout(self.inner_layout)

    def _add_advanced_options(self):
        self.tile_x_label = QLabel("Number X tiles:")
        self.tile_x_label.setToolTip(
            format_tooltip(
                """
Number of tiles to split the image into in the X dimension. 'auto' allows Nextflow to decide an appropriate split.
"""
            )
        )
        self.tile_x = QSpinBox(minimum=0, maximum=100, value=0)
        self.tile_x.setSpecialValueText("auto")
        self.tile_x.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.tile_y_label = QLabel("Number Y tiles:")
        self.tile_y_label.setToolTip(
            format_tooltip(
                """
Number of tiles to split the image into in the Y dimension. 'auto' allows Nextflow to decide an appropriate split.
"""
            )
        )
        self.tile_y = QSpinBox(minimum=0, maximum=100, value=0)
        self.tile_y.setSpecialValueText("auto")
        self.tile_y.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.tile_z_label = QLabel("Number Z tiles:")
        self.tile_z_label.setToolTip(
            format_tooltip(
                """
Number of tiles to split the image into in the Z dimension. 'auto' allows Nextflow to decide an appropriate split.
"""
            )
        )
        self.tile_z = QSpinBox(minimum=0, maximum=100, value=0)
        self.tile_z.setSpecialValueText("auto")
        self.tile_z.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.advanced_layout.addWidget(self.tile_x_label, 0, 0, 1, 1)
        self.advanced_layout.addWidget(self.tile_x, 0, 1, 1, 1)
        self.advanced_layout.addWidget(self.tile_y_label, 1, 0, 1, 1)
        self.advanced_layout.addWidget(self.tile_y, 1, 1, 1, 1)
        self.advanced_layout.addWidget(self.tile_z_label, 2, 0, 1, 1)
        self.advanced_layout.addWidget(self.tile_z, 2, 1, 1, 1)

        self.overlap_x_label = QLabel("Overlap X:")
        self.overlap_x_label.setToolTip(
            format_tooltip(
                "Fraction of overlap between tiles in the X dimension."
            )
        )
        self.overlap_x = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.0)
        self.overlap_x.setSingleStep(0.05)
        self.overlap_x.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.overlap_y_label = QLabel("Overlap Y:")
        self.overlap_y_label.setToolTip(
            format_tooltip(
                "Fraction of overlap between tiles in the Y dimension."
            )
        )
        self.overlap_y = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.0)
        self.overlap_y.setSingleStep(0.05)
        self.overlap_y.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.overlap_z_label = QLabel("Overlap Z:")
        self.overlap_z_label.setToolTip(
            format_tooltip(
                "Fraction of overlap between tiles in the Z dimension."
            )
        )
        self.overlap_z = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.0)
        self.overlap_z.setSingleStep(0.05)
        self.overlap_z.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.advanced_layout.addWidget(self.overlap_x_label, 3, 0, 1, 1)
        self.advanced_layout.addWidget(self.overlap_x, 3, 1, 1, 1)
        self.advanced_layout.addWidget(self.overlap_y_label, 4, 0, 1, 1)
        self.advanced_layout.addWidget(self.overlap_y, 4, 1, 1, 1)
        self.advanced_layout.addWidget(self.overlap_z_label, 5, 0, 1, 1)
        self.advanced_layout.addWidget(self.overlap_z, 5, 1, 1, 1)

        # Connect all the spinboxes to the same function
        self.tile_x.valueChanged.connect(self.update_tile_size)
        self.tile_y.valueChanged.connect(self.update_tile_size)
        self.tile_z.valueChanged.connect(self.update_tile_size)
        self.overlap_x.valueChanged.connect(self.update_tile_size)
        self.overlap_y.valueChanged.connect(self.update_tile_size)
        self.overlap_z.valueChanged.connect(self.update_tile_size)

        self.tile_size_label = QLabel("No image layers found!")
        self.tile_size_label.setToolTip(
            "Tile size based on currently selected image and tile settings above."
        )
        self.advanced_layout.addWidget(self.tile_size_label, 6, 0, 1, 2)

        # Add post-processing options
        self.postprocess_btn = QCheckBox("Re-label output")
        self.postprocess_btn.setChecked(True)
        self.postprocess_btn.setToolTip(
            format_tooltip(
                """
If checked, the model output will be re-labelled using connected components to create consistency across slices.
            """
            )
        )
        self.advanced_layout.addWidget(self.postprocess_btn, 7, 0, 1, 2)
        # Add threshold for IoU SAM post-processing
        self.iou_thresh_label = QLabel("IoU threshold (SAM only):")
        self.iou_thresh_label.setToolTip(
            format_tooltip(
                """
Threshold for the Intersection over Union (IoU) metric used in the SAM post-processing step.
            """
            )
        )
        self.iou_thresh = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.8)
        self.iou_thresh.setSingleStep(0.01)
        self.iou_thresh.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.advanced_layout.addWidget(self.iou_thresh_label, 8, 0, 1, 1)
        self.advanced_layout.addWidget(self.iou_thresh, 8, 1, 1, 1)

        # Run the function to update the tile size label to get initial value
        self.update_tile_size(val=None, clear_label=False)

    def on_toggle_advanced(self):
        if self.advanced_box.isChecked():
            self.advanced_widget.setVisible(True)
            self.advanced_box.setText(" ▼ Advanced Options")
        else:
            self.advanced_widget.setVisible(False)
            self.advanced_box.setText(" ▶ Advanced Options")

    def store_img_paths(self, img_paths: list[Path]):
        """
        Writes the provided image paths to a file to pass into Nextflow.

        TODO: May be subject to complete rewrite with dask/zarr
        """
        # Create container for metadata
        output = defaultdict(list)
        # Create container for knowing what images to track progress of
        self.progress_dict = {}
        # Counter for number of substacks (equivalent to number of submitted jobs!)
        total_substacks = 0
        # Extract inputted stack size
        stack_size = (
            (
                "auto"
                if self.tile_x.value() == self.tile_x.minimum()
                else self.tile_x.value()
            ),
            (
                "auto"
                if self.tile_y.value() == self.tile_y.minimum()
                else self.tile_y.value()
            ),
            (
                "auto"
                if self.tile_z.value() == self.tile_z.minimum()
                else self.tile_z.value()
            ),
        )
        # Convert into Stack namedtuple
        stack_size = Stack(
            height=stack_size[0], width=stack_size[1], depth=stack_size[2]
        )
        # Extract overlap fraction
        overlap_frac = Stack(
            height=round(self.overlap_x.value(), 2),
            width=round(self.overlap_y.value(), 2),
            depth=round(self.overlap_z.value(), 2),
        )
        # Extract info from each image
        for img_path in img_paths:
            # Get the mask layer name
            layer = self.parent.viewer.layers[img_path.stem]
            # Get the number of slices, channels, height, and width
            H, W, num_slices, channels = get_img_dims(layer, img_path)
            output["img_path"].append(str(img_path))
            output["num_slices"].append(num_slices)
            output["height"].append(H)
            output["width"].append(W)
            output["channels"].append(channels)
            # Initialise the progress dict
            self.progress_dict[img_path.stem] = 0
            # Get the actual stack size
            img_shape = Stack(
                height=H, width=W, depth=num_slices, channels=channels
            )
            num_substacks, eff_shape = calc_num_stacks(
                image_shape=img_shape,
                req_stacks=stack_size,
                overlap_fraction=overlap_frac,
            )
            # Get the number of substacks
            _, num_substacks, _ = generate_stack_indices(
                image_shape=img_shape,
                num_substacks=num_substacks,
                overlap_fraction=overlap_frac,
                eff_shape=eff_shape,
            )
            total_substacks += num_substacks
        # Convert to a DataFrame and save
        df = pd.DataFrame(output)
        df.to_csv(self.img_list_fpath, index=False)
        # Store the total number of jobs
        self.total_substacks = total_substacks

    def check_inference(self):
        """
        Checks that all the necessary parameters are set for inference.

        Checks that:
        - A task has been selected
        - A model has been selected
        - Data has been selected
        """
        if self.parent.selected_task is None:
            raise ValueError("No task/organelle selected!")
        if self.parent.selected_model is None:
            raise ValueError("No model selected!")
        if len(self.parent.subwidgets["data"].image_path_dict) == 0:
            raise ValueError("No data selected!")

    def setup_inference(self, nxf_params: Optional[dict] = None):
        """
        Runs the inference pipeline in Nextflow.

        `nxf_params` is a dict containing everything that Nextflow needs at the command line.
        """
        # Store the selected task, model, and variant for execution
        self.parent.executed_task = self.parent.selected_task
        self.parent.executed_model = self.parent.selected_model
        self.parent.executed_variant = self.parent.selected_variant
        # Set the starting Nextflow command
        nxf_cmd = self.nxf_base_cmd + f"run {self.nxf_repo} -latest"
        # nxf_params can only be given when used standalone, which is rare
        if nxf_params is not None:
            return nxf_cmd, nxf_params  # FIXME: Returns diff number variables
        # Construct the Nextflow params if not given
        parent = self.parent
        # Get the model config path
        config_path = parent.subwidgets["model"].get_model_config()
        # Construct the proper mask directory path
        self.mask_dir_path = (
            self.nxf_store_dir
            / f"{parent.executed_model}"
            / f"{sanitise_name(parent.executed_variant)}_masks"
        )
        # Construct the params to be given to Nextflow
        nxf_params = {}
        nxf_params["root_dir"] = str(self.nxf_base_dir)
        nxf_params["img_dir"] = str(self.img_list_fpath)
        nxf_params["model"] = parent.selected_model
        nxf_params["model_config"] = str(config_path)
        nxf_params["model_type"] = sanitise_name(parent.executed_variant)
        nxf_params["task"] = parent.executed_task
        # Extract the model checkpoint location and location type
        model_task = parent.subwidgets["model"].model_version_tasks[
            (
                parent.executed_task,
                parent.executed_model,
                parent.executed_variant,
            )
        ]
        # Location type determined from registry schema
        nxf_params["model_chkpt_type"] = model_task.location_type
        if model_task.location_type == "url":
            # This parses the URL to get the root filename which we'll use
            res = urlparse(model_task.location)
            nxf_params["model_chkpt_loc"] = model_task.location
            nxf_params["model_chkpt_fname"] = Path(res.path).name
        elif model_task.location_type == "file":
            res = Path(model_task.location)
            nxf_params["model_chkpt_loc"] = res.parent
            nxf_params["model_chkpt_fname"] = res.name
        # Extract the tiles and overlap
        # Special text is ignored by default, so need to convert
        num_substacks = []
        num_substacks.append(
            "auto"
            if self.tile_x.value() == self.tile_x.minimum()
            else self.tile_x.value()
        )
        num_substacks.append(
            "auto"
            if self.tile_y.value() == self.tile_y.minimum()
            else self.tile_y.value()
        )
        num_substacks.append(
            "auto"
            if self.tile_z.value() == self.tile_z.minimum()
            else self.tile_z.value()
        )
        # Nextflow needs a comma-separated string for multiple values
        nxf_params["num_substacks"] = ",".join(map(str, num_substacks))
        nxf_params["overlap"] = (
            f"{round(self.overlap_x.value(), 2)},{round(self.overlap_y.value(), 2)},{round(self.overlap_z.value(), 2)}"
        )
        nxf_params["iou_threshold"] = round(self.iou_thresh.value(), 2)
        # Get the preprocessing options
        nxf_params["preprocess"] = parent.subwidgets[
            "preprocess"
        ].extract_options()
        # Now have everything for the run hash
        self.parent.get_run_hash(nxf_params)
        # No need to check if we are ovewriting
        if self.overwrite_btn.isChecked():
            proceed = True
            load_paths = []
            img_paths = self.parent.subwidgets["data"].image_path_dict.values()
            # Delete data in mask layers if present
            for img_path in img_paths:
                # Get the mask layer name
                layer_name = self.parent._get_mask_layer_name(
                    Path(img_path).stem
                )
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
            # Delete current masks
            for mask_path in self.mask_dir_path.glob("*.npy"):
                mask_path.unlink()
        # Check if we already have all the masks
        else:
            proceed, img_paths, load_paths = self.parent.check_masks()
        # If some masks need loading, load them
        if load_paths:
            self.parent.create_mask_layers(img_paths=load_paths)
        # If we already have all the masks, don't run the pipeline
        if not proceed:
            show_info(
                f"Masks already exist for all files for segmenting {TASK_NAMES[parent.executed_task]} with {parent.executed_model} ({parent.executed_variant})!"
            )
            # Enable the export button as all masks available
            self.export_masks_btn.setEnabled(True)
            # Otherwise, until importing is fully sorted, the user just gets a notification and that's it
            return nxf_cmd, nxf_params, proceed, img_paths
        else:
            # Start the watcher for the mask files
            self.parent.watch_mask_files()
            return nxf_cmd, nxf_params, proceed, img_paths

    def setup_finetuning(self):
        """
        Runs the finetuning pipeline in Nextflow.
        """
        raise NotImplementedError

    def run_pipeline(self):
        if "data" not in self.parent.subwidgets:
            raise ValueError("Cannot run pipeline without data widget!")
        # Store the image paths
        self.image_path_dict = self.parent.subwidgets["data"].image_path_dict
        # Ensure the pipeline is valid
        assert (
            self.pipeline in self.pipelines.keys()
        ), f"Pipeline {self.pipeline} not found!"
        # Do the initial checks
        if self.pipelines[self.pipeline]["check"] is not None:
            self.pipelines[self.pipeline]["check"]()
        else:
            raise NotImplementedError(
                f"Pipeline {self.pipeline} check function not implemented!"
            )
        if self.all_loaded is False:
            # Check whether layers already existed when plugin started, and if all were loaded
            if not (
                len(self.image_path_dict) > 0
                and self.parent.subwidgets["data"].existing_loaded
            ):
                show_info("Not all images have loaded, please wait...")
                return
        # Get the pipeline-specific stuff
        nxf_cmd, nxf_params, proceed, img_paths = self.pipelines[
            self.pipeline
        ]["setup"]()
        # Don't run the pipeline if no green light given
        if not proceed:
            return
        # Store plugin settings for future sessions
        self.parent.store_settings()
        # Store the image paths
        self.store_img_paths(img_paths=img_paths)
        # Add custom work directory
        if self.nxf_work_dir is not None:
            nxf_cmd += f" -w {self.nxf_work_dir}"
        # Add the selected profile to the command
        nxf_cmd += f" -profile {self.nxf_profile_box.currentText()}"
        # Add postprocessing flag
        nxf_params["postprocess"] = self.postprocess_btn.isChecked()
        # Add the Nextflow parameter hash to the command
        nxf_params["param_hash"] = self.parent.run_hash
        # Save the Nextflow parameters to a YAML file
        nxf_params_fpath = (
            self.nxf_store_dir / f"nxf_params_{self.parent.run_hash}.yml"
        )
        with open(nxf_params_fpath, "w") as f:
            yaml.dump(nxf_params, f)
        # Add params-file to nxf command
        nxf_cmd += f" -params-file {nxf_params_fpath}"

        @thread_worker(
            connect={
                "started": self._pipeline_start,
                "returned": self._pipeline_finish,
                "errored": self._pipeline_fail,
            }
        )
        def _run_pipeline(nxf_cmd: str):
            # Run the command
            self.process = subprocess.Popen(
                nxf_cmd, shell=True, cwd=Path.home()
            )
            self.process.wait()
            # Check if the process was successful
            if self.process.returncode != 0:
                raise RuntimeError

        # Run the pipeline
        _run_pipeline(nxf_cmd)

    def _reset_btns(self):
        """
        Resets the buttons to their original state.
        """
        self.nxf_run_btn.setText("Run Pipeline!")
        self.nxf_run_btn.setEnabled(True)
        self.export_masks_btn.setEnabled(True)
        self._remove_cancel_btn()

    def _pipeline_start(self):
        # Add a notification that the pipeline has started
        show_info("Pipeline started!")
        # Modify buttons during run
        self.export_masks_btn.setEnabled(False)
        # Disable the button to avoid issues
        # TODO: Enable multiple job execution, may require -bg flag?
        self.nxf_run_btn.setEnabled(False)
        # Update the button to signify it's running
        self.nxf_run_btn.setText("Running Pipeline...")
        self.init_progress_bar()
        # Add a cancel pipeline button
        idx = self.inner_widget.layout().indexOf(self.nxf_run_btn)
        row, col, rowspan, colspan = (
            self.inner_widget.layout().getItemPosition(idx)
        )
        self.orig_colspan = colspan
        self.cancel_btn = QPushButton("Cancel Pipeline")
        self.cancel_btn.clicked.connect(self.cancel_pipeline)
        self.cancel_btn.setToolTip("Cancel the currently running pipeline.")
        new_colspan = colspan // 2 if colspan > 1 else 1
        self.inner_widget.layout().addWidget(
            self.nxf_run_btn, row, col, rowspan, new_colspan
        )
        self.inner_widget.layout().addWidget(
            self.cancel_btn, row, col + new_colspan, rowspan, new_colspan
        )

    def _pipeline_finish(self):
        # Add a notification that the pipeline has finished
        show_info("Pipeline finished!")
        self._reset_btns()
        # When finished, insert all '_all' masks to ensure everything is correct
        self.parent.insert_final_masks()
        # Ensure progress bar is at 100%
        self.pbar.setValue(self.total_substacks)

    def _pipeline_fail(self, exc):
        show_info("Pipeline failed! See terminal for details")
        print(exc)
        self._reset_btns()
        # Deactivate file watcher
        if hasattr(self.parent, "watcher_enabled"):
            print("Deactivating watcher...")
            self.parent.watcher_enabled = False

    def _remove_cancel_btn(self):
        # Remove the cancel pipeline button
        self.inner_widget.layout().removeWidget(self.cancel_btn)
        self.cancel_btn.setParent(None)
        idx = self.inner_widget.layout().indexOf(self.nxf_run_btn)
        row, col, rowspan, _ = self.inner_widget.layout().getItemPosition(idx)
        self.inner_widget.layout().addWidget(
            self.nxf_run_btn, row, col, rowspan, self.orig_colspan
        )

    def on_click_base_dir(self):
        """
        Callback for when the base directory button is clicked. Opens a dialog to select a directory to save the masks to.
        """
        base_dir = QFileDialog.getExistingDirectory(
            self, caption="Select directory to store cache", directory=None
        )
        # Skip if no directory selected
        if base_dir == "":
            return
        # Replace any spaces, makes everything else easier
        new_dir_name = Path(base_dir).name.replace(" ", "_")
        base_dir = Path(base_dir).parent / new_dir_name
        # Update the text
        self.nxf_dir_text.setText(str(base_dir))
        # Update the base directory and Nextflow command
        self.setup_nxf_dir_cmd(base_dir=base_dir)

    def init_progress_bar(self):
        # Set the values of the Qt progress bar
        self.pbar.setRange(0, self.total_substacks)
        self.pbar.setValue(0)
        # Initialise the tqdm progress bar to monitor time
        self.tqdm_pbar = tqdm.tqdm(total=self.total_substacks)
        # Reset the label
        self.pbar_label.setText("Progress: [--:--]")

    def update_progress_bar(self):
        # Update the progress bar to the current number of slices
        curr_slices = sum(self.progress_dict.values())
        self.pbar.setValue(curr_slices)
        self.tqdm_pbar.update(curr_slices - self.tqdm_pbar.n)
        # Update the label
        elapsed = self.tqdm_pbar.format_dict["elapsed"]
        rate = (
            self.tqdm_pbar.format_dict["rate"]
            if self.tqdm_pbar.format_dict["rate"]
            else 1
        )
        remaining = (self.tqdm_pbar.total - self.tqdm_pbar.n) / rate
        self.pbar_label.setText(
            f"Progress: [{self.tqdm_pbar.format_interval(elapsed)}<{self.tqdm_pbar.format_interval(remaining)}]"
        )

    def reset_progress_bar(self):
        # Set the values of the Qt progress bar
        self.pbar.setValue(0)
        # Close the tqdm progress bar
        self.tqdm_pbar.close()
        # Reset the label
        self.pbar_label.setText("Progress: [--:--]")

    def on_click_import(self):
        """
        Callback for when the import button is clicked. Opens a dialog to select mask files to import.

        Expectation is that these come from the Nextflow and are therefore .npy files. For anything external, they can be added to Napari as normal.

        TODO: Current disabled, as arbitrary import makes it harder to allow partial pipeline running.
        """
        fnames, _ = QFileDialog.getOpenFileNames(
            self,
            caption="Select mask files to import",
            directory=str(Path.home()),
            filter="Numpy files (*.npy)",  # NOTE: Will need to change when moving away from numpy
        )
        for fname in fnames:
            mask_arr = np.load(fname, allow_pickle=True)
            self.viewer.add_labels(
                mask_arr,
                name=Path(fname).stem.replace("_all", ""),
                visible=True,
                opacity=0.5,
            )

    def on_click_export(self):
        """
        Callback for when the export button is clicked. Opens a dialog to select a directory to save the masks to.
        """
        export_dir = QFileDialog.getExistingDirectory(
            self, caption="Select directory to save masks", directory=None
        )
        # Extract the data from each of the layers, and save the result in the given folder
        # NOTE: Will also need adjusting for the dask/zarr rewrite
        if self.selected_mask_layers:
            count = 0
            for mask_layer in self.selected_mask_layers:
                # Get the name of the mask layer as root for the filename
                fname = f"{mask_layer.name}"
                # Check if we are binarising
                if self.export_binary_check.isChecked():
                    mask_data = self._binarise_mask(mask_layer)
                    fname += "_binarised"
                else:
                    mask_data = mask_layer.data
                # Get the extension & add to fname
                ext = self.export_format_dropdown.currentText().strip(".")
                fname += f".{ext}"
                if ext == "npy":
                    np.save(
                        Path(export_dir) / fname,
                        mask_data,
                    )
                elif ext == "tiff":
                    skimage.io.imsave(
                        Path(export_dir) / fname,
                        mask_data,
                        plugin="tifffile",
                    )
                count += 1
            show_info(f"Exported {count} mask files to {export_dir}!")
        else:
            show_info("No mask layers found!")

    def cancel_pipeline(self):
        # Trigger Nextflow to cancel the pipeline
        self.process.send_signal(subprocess.signal.SIGTERM)
        # Reset the progress bar
        self.reset_progress_bar()

    def _binarise_mask(self, mask_layer):
        """
        Binarises the given mask layer.
        """
        return (mask_layer.data).astype(bool).astype(np.uint8) * 255

    def update_tile_size(
        self, val: Union[int, float], clear_label: bool = False
    ):
        """
        Callback for when the tile size spinboxes are updated.
        """
        # Get the stack size
        # FIXME: Pattern repeated 3 times in this script, abstract?
        # Extract inputted stack size
        stack_size = (
            (
                "auto"
                if self.tile_x.value() == self.tile_x.minimum()
                else self.tile_x.value()
            ),
            (
                "auto"
                if self.tile_y.value() == self.tile_y.minimum()
                else self.tile_y.value()
            ),
            (
                "auto"
                if self.tile_z.value() == self.tile_z.minimum()
                else self.tile_z.value()
            ),
        )
        # Convert into Stack namedtuple
        stack_size = Stack(
            height=stack_size[0], width=stack_size[1], depth=stack_size[2]
        )
        # Extract overlap fraction
        overlap_frac = Stack(
            height=round(self.overlap_x.value(), 2),
            width=round(self.overlap_y.value(), 2),
            depth=round(self.overlap_z.value(), 2),
        )
        # Get the relevant image shape
        # First check if we have any layers selected
        if len(self.viewer.layers.selection) >= 1:
            layers = self.viewer.layers.selection
        # Otherwise get all layers
        else:
            layers = self.viewer.layers
        # Filter down to only Image layers
        layers = [
            layer for layer in layers if isinstance(layer, napari.layers.Image)
        ]
        # Check if we have any image layers
        if len(layers) == 0 or clear_label:
            self.tile_size_label.setText("No image layers found!")
            return
        # Otherwise just take the first one
        H, W, num_slices, channels = get_img_dims(layers[0])
        img_shape = Stack(
            height=H, width=W, depth=num_slices, channels=channels
        )
        # Get the actual stack size
        num_substacks, eff_shape = calc_num_stacks(
            image_shape=img_shape,
            req_stacks=stack_size,
            overlap_fraction=overlap_frac,
        )
        # Get the number of substacks
        _, num_substacks, stack_size_px = generate_stack_indices(
            image_shape=img_shape,
            num_substacks=num_substacks,
            overlap_fraction=overlap_frac,
            eff_shape=eff_shape,
        )

        self.tile_size_label.setText(
            format_tooltip(
                f"Substack size: {stack_size_px.depth} slice{'s' if stack_size_px.depth > 1 else ''}, {stack_size_px.height}px x {stack_size_px.width}px for each of the {num_substacks} jobs to submit (for the selected image).",
                width=40,
            )
        )
