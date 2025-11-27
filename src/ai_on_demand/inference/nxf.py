from collections import defaultdict
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Optional, Union
from urllib.parse import urlparse

from aiod_registry import TASK_NAMES
import napari
import paramiko
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
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
    QGroupBox,
    QMessageBox,
    QLineEdit,
)
import tqdm
import yaml

from ai_on_demand.utils import sanitise_name, format_tooltip, get_img_dims
from ai_on_demand.widget_classes import SubWidget
import aiod_utils.preprocess
from aiod_utils.stacks import generate_stack_indices, calc_num_stacks, Stack


class NxfWidget(SubWidget):
    _name = "nxf"

    config_ready = qtpy.QtCore.Signal()

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

        self.nxf_cmd = None
        self.nxf_params = None

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
                if Path(settings["base_dir"]).exists():
                    nxf_base_dir = Path(settings["base_dir"])
                # in the case where user has unmounted a drive (e.g. remote server drive for ssh pipeline)
                else:
                    print(
                        f"Warning: Could not access {settings['base_dir']}, falling back to default cache directory."
                    )
                    nxf_base_dir = Path.home() / ".nextflow" / "aiod"
                    nxf_base_dir.mkdir(parents=True, exist_ok=True)
                self.nxf_dir_text.setText(str(nxf_base_dir))
                # Update the base directory and Nextflow command
                self.setup_nxf_dir_cmd(base_dir=Path(nxf_base_dir))

    def get_settings(self) -> dict:
        """
        Get the settings for the plugin to store for future sessions.
        """
        settings = {
            "base_dir": str(self.nxf_base_dir),
            "profile": self.nxf_profile_box.currentText(),
        }
        return settings

    def get_config_params(self, params):
        widget_config = {
            "base_dir": str(self.nxf_base_dir),
            "profile": self.nxf_profile_box.currentText(),
            "advanced_options": {
                "num_substacks": params.get("num_substacks"),
                "overlap": params.get("overlap"),
                "iou_threshold": params.get("iou_threshold"),
            },
            "ssh_settings": {
                "active": self.ssh_box.isChecked(),
                "hostname": self.hostname.text(),
                "target_node": self.target_node.text(),
                "username": self.username.text(),
                "remote_path_prefix": self.remote_path_prefix.text(),
                "mounted_path_prefix": self.mounted_path_prefix.text(),
                "ssh_key_path": self.ssh_key_path,
            },
        }
        return widget_config

    def load_config(self, config):
        profile_index = self.nxf_profile_box.findText(config["profile"])
        if profile_index != -1:
            self.nxf_profile_box.setCurrentIndex(profile_index)
        base_dir = config["base_dir"]
        if self.nxf_dir_text.text() != base_dir:
            self.nxf_dir_text.setText(base_dir)
            self.setup_nxf_dir_cmd(base_dir=Path(base_dir))

        # loading advanced options
        adv = config["advanced_options"]
        num_substacks = adv.get("num_substacks")
        tile_boxes = [self.tile_x, self.tile_y, self.tile_z]
        for box, val in zip(tile_boxes, num_substacks.split(",")):
            if val == "auto":
                box.setValue(-1)
            else:
                box.setValue(int(val))

        overlap_str = adv.get("overlap")
        overlap = [float(i) for i in overlap_str.split(",")]

        self.overlap_x.setValue(overlap[0])
        self.overlap_y.setValue(overlap[1])
        self.overlap_z.setValue(overlap[2])

        self.iou_thresh.setValue(float(adv.get("iou_threshold")))

        # loading ssh section
        ssh_settings = config["ssh_settings"]
        if ssh_settings["active"]:
            self.ssh_box.setChecked(True)
        else:
            self.ssh_box.setChecked(False)
        self.hostname.setText(ssh_settings["hostname"])
        self.target_node.setText(ssh_settings["target_node"])
        self.username.setText(ssh_settings["username"])
        self.remote_path_prefix.setText(ssh_settings["remote_path_prefix"])
        self.mounted_path_prefix.setText(ssh_settings["mounted_path_prefix"])
        self.ssh_key_path = ssh_settings["ssh_key_path"]
        self.ssh_key_label.setText(f"SSH Key: {ssh_settings['ssh_key_path']}")

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

    def _create_ssh_box(self):
        """
        Builds the SSH settings UI and attaches it to self.inner_layout.
        Extracted from create_box to keep the main layout method concise.
        """
        self.ssh_box = QGroupBox("ssh settings")
        self.ssh_box.setToolTip(
            "Settings for running nextflow pipeline via SSH"
        )
        self.ssh_box.setCheckable(True)
        self.ssh_box.setChecked(False)

        self.ssh_layout = QGridLayout()
        self.ssh_box.setLayout(self.ssh_layout)

        self.hostname = QLineEdit(placeholderText="Enter hostname here...")
        self.target_node = QLineEdit(
            placeholderText="Enter target_node here..."
        )
        self.username = QLineEdit(placeholderText="Enter username here...")
        self.passphrase_input = QLineEdit(
            placeholderText="Enter passphrase here..."
        )
        self.passphrase_input.setEchoMode(QLineEdit.Password)
        self.remote_path_prefix = QLineEdit(placeholderText="e.g. /nemo/stp/")
        self.mounted_path_prefix = QLineEdit(placeholderText="e.g. /Volumes/")
        self.command_prepend = QLineEdit(
            placeholderText="Enter command prepend here..."
        )

        # SSH key section
        self.info_btn = QPushButton("i")
        self.info_btn.setFixedWidth(30)
        self.info_btn.setToolTip("Help I don't know which ssh key to pick!")
        self.info_btn.clicked.connect(self._show_ssh_info)

        self.ssh_key_path = ""
        self.ssh_key_label = QLabel("SSH Key: Not selected")
        self.locate_key_btn = QPushButton("Locate SSH Key")
        self.locate_key_btn.clicked.connect(self._locate_ssh_key)

        # SSH Layout
        self.ssh_layout.addWidget(QLabel("Hostname:"), 0, 0)
        self.ssh_layout.addWidget(self.hostname, 0, 1, 1, 2)

        self.ssh_layout.addWidget(QLabel("Target node:"), 1, 0)
        self.ssh_layout.addWidget(self.target_node, 1, 1, 1, 2)

        self.ssh_layout.addWidget(QLabel("Username:"), 2, 0)
        self.ssh_layout.addWidget(self.username, 2, 1, 1, 2)

        self.ssh_layout.addWidget(QLabel("Passphrase:"), 3, 0)
        self.ssh_layout.addWidget(self.passphrase_input, 3, 1, 1, 2)

        self.ssh_layout.addWidget(QLabel("Remote path prefix:"), 4, 0)
        self.ssh_layout.addWidget(self.remote_path_prefix, 4, 1, 1, 2)

        self.ssh_layout.addWidget(QLabel("Mounted path prefix:"), 5, 0)
        self.ssh_layout.addWidget(self.mounted_path_prefix, 5, 1, 1, 2)

        self.ssh_layout.addWidget(QLabel("Command prepend:"), 6, 0)
        self.ssh_layout.addWidget(self.command_prepend, 6, 1, 1, 2)

        self.ssh_layout.addWidget(self.ssh_key_label, 7, 0, 1, 3)
        self.ssh_layout.addWidget(self.locate_key_btn, 8, 0, 1, 2)
        self.ssh_layout.addWidget(self.info_btn, 8, 2)

        self.inner_layout.addWidget(self.ssh_box, 2, 0, 1, 2)

    def create_box(self, variant: Optional[str] = None):
        # Create box for the cache settings
        self.cache_box = QGroupBox("Cache Settings")
        self.cache_box.setToolTip(
            format_tooltip(
                "Settings for the AIoD/Nextflow cache for storing models and results."
            )
        )
        self.cache_layout = QGridLayout()
        self.cache_box.setLayout(self.cache_layout)
        # Create the option for selecting base directory
        self.nxf_dir_label = QLabel("Base directory:")
        base_dir_tooltip = "Select the base directory to store the Nextflow cache (i.e. all models & results) in."
        self.nxf_dir_label.setToolTip(format_tooltip(base_dir_tooltip))
        self.nxf_dir_text = QLabel(str(self.nxf_base_dir))
        self.nxf_dir_text.setWordWrap(True)
        self.nxf_dir_text.setToolTip(
            format_tooltip("The selected base directory.")
        )
        self.nxf_dir_text.setMaximumWidth(400)
        # Button to change the base directory
        self.nxf_dir_btn = QPushButton("Change")
        self.nxf_dir_btn.clicked.connect(self.on_click_base_dir)
        self.nxf_dir_btn.setToolTip(format_tooltip(base_dir_tooltip))
        # Button to inspect the base directory/cache
        self.nxf_dir_inspect_btn = QPushButton("Inspect cache")
        self.nxf_dir_inspect_btn.clicked.connect(self.on_click_inspect_cache)
        self.nxf_dir_inspect_btn.setToolTip(
            format_tooltip(
                """
Open the base directory in the file explorer to inspect the cache.

Note that 'opening' won't do anything, this is just to see what files are present.
"""
            )
        )
        # Button to clear the cache
        self.nxf_dir_clear_btn = QPushButton("Clear cache")
        self.nxf_dir_clear_btn.clicked.connect(self.on_click_clear_cache)
        self.nxf_dir_clear_btn.setToolTip(
            format_tooltip(
                "Clear the cache of all models and results. WARNING: This will remove all models and results from the cache."
            )
        )

        # Layout all the cache settings
        self.cache_layout.addWidget(self.nxf_dir_label, 0, 0, 1, 2)
        self.cache_layout.addWidget(self.nxf_dir_text, 0, 2, 1, 3)
        self.cache_layout.addWidget(self.nxf_dir_btn, 0, 5, 1, 1)
        self.cache_layout.addWidget(self.nxf_dir_inspect_btn, 1, 0, 1, 3)
        self.cache_layout.addWidget(self.nxf_dir_clear_btn, 1, 3, 1, 3)

        # Add the cache box to the main layout
        self.inner_layout.addWidget(self.cache_box, 0, 0, 1, 2)

        # Create box for the cache settings
        self.pipeline_box = QGroupBox("Pipeline Settings")
        self.pipeline_box.setToolTip(
            format_tooltip("Settings for the Segment-Flow pipeline itself.")
        )
        self.pipeline_layout = QGridLayout()
        self.pipeline_box.setLayout(self.pipeline_layout)

        # Create a drop-down box to select the execution profile
        self.nxf_profile_label = QLabel("Execution profile:")
        self.nxf_profile_label.setToolTip(
            format_tooltip("Select the execution profile to use.")
        )
        self.nxf_profile_box = QComboBox()
        # Get the available profiles from config dir
        config_dir = Path(__file__).parent.parent / "Segment-Flow" / "profiles"
        avail_confs = [str(i.stem) for i in config_dir.glob("*.conf")]
        avail_confs.sort()
        if len(avail_confs) == 0:
            raise FileNotFoundError(
                f"No Nextflow profiles found in {config_dir}!"
            )
        self.nxf_profile_box.addItems(avail_confs)
        self.nxf_profile_box.setFocusPolicy(
            qtpy.QtCore.Qt.FocusPolicy.StrongFocus
        )
        self.pipeline_layout.addWidget(self.nxf_profile_label, 0, 0)
        self.pipeline_layout.addWidget(self.nxf_profile_box, 0, 1)

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
        self.pipeline_layout.addWidget(self.overwrite_btn, 1, 0, 1, 1)

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
        self.advanced_layout.setContentsMargins(4, 0, 4, 0)
        self.options_widget.setLayout(self.options_layout)
        self.pipeline_layout.addWidget(self.options_widget, 3, 0, 1, 2)

        self.inner_layout.addWidget(self.pipeline_box, 1, 0, 1, 2)

        self._create_ssh_box()

        # Create a button to navigate to a directory to take images from
        self.nxf_run_btn = QPushButton("Run Pipeline!")
        self.nxf_run_btn.clicked.connect(self.run_pipeline)
        self.nxf_run_btn.setToolTip(
            format_tooltip(
                "Run the pipeline with the chosen organelle(s), model, and images."
            )
        )
        self.inner_layout.addWidget(self.nxf_run_btn, 3, 0, 1, 2)

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
        self.postprocess_btn.setChecked(False)
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
            # Need to take account for multiple runs due to preprocessing
            relevant_runs = [
                i
                for i in self.parent.img_mask_info
                if i["img_path"].stem == img_path.stem
            ]
            for d in relevant_runs:
                # Get the shape after preprocessing (if any)
                if d["prep_set"] is None:
                    final_shape = Stack(
                        height=H, width=W, depth=num_slices, channels=channels
                    )
                else:
                    output_shape = aiod_utils.preprocess.get_output_shape(
                        d["prep_set"], input_shape=(num_slices, H, W)
                    )
                    final_shape = Stack(
                        height=output_shape[1],
                        width=output_shape[2],
                        depth=output_shape[0],
                        channels=channels,
                    )
                # Calculate the number of substacks
                num_substacks, eff_shape = calc_num_stacks(
                    image_shape=final_shape,
                    req_stacks=stack_size,
                    overlap_fraction=overlap_frac,
                )
                # Get the number of substacks
                _, num_substacks, _ = generate_stack_indices(
                    image_shape=final_shape,
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
            nxf_params["model_chkpt_loc"] = str(res.parent)
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
        ].get_all_options()
        # Now have everything for the run hash
        parent.get_run_hash(nxf_params)
        # If overwriting, delete existing mask layers and files
        if self.overwrite_btn.isChecked():
            proceed = True
            load_paths = []
            img_paths = self.parent.subwidgets["data"].image_path_dict.values()
            # Extract layer names, considering current preprocessing sets etc.
            parent.get_img_mask_preps()
            all_layer_names = [i["layer_name"] for i in parent.img_mask_info]
            # Delete data in mask layers if present
            for layer_name in all_layer_names:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
            # Delete masks that match determined layer names (this'll remove partial and full masks, if present)
            for mask_path in self.mask_dir_path.glob("*.rle"):
                for layer_name in all_layer_names:
                    if layer_name in mask_path.stem:
                        mask_path.unlink()
                        break
        # Check if we already have all the masks
        else:
            proceed, img_paths, load_paths = self.parent.check_masks()
        # If some masks need loading, load them
        if load_paths:
            self.nxf_run_btn.setEnabled(False)
            self.nxf_run_btn.setText("Loading already-run masks...")
            self.parent.create_mask_layers(img_paths=load_paths)
        # Reset the run button after loading just in case
        self.nxf_run_btn.setText("Run Pipeline!")
        self.nxf_run_btn.setEnabled(True)
        # If we already have all the masks, don't run the pipeline
        if not proceed:
            msg = f"Masks already exist for all files for segmenting {TASK_NAMES[parent.executed_task]} with {parent.executed_model} ({parent.executed_variant})!"
            if self.parent.run_hash is not None:
                msg += f" (Hash: {self.parent.run_hash[:8]})"
            show_info(msg)
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
            # We use shlex to ensure the command is properly escaped
            # We use shell=False to avoid shell injection issues
            # We use -l to ensure the command is run in a login shell, avoiding conda issues
            self.process = subprocess.Popen(
                ["/bin/sh", "-l", "-c"] + shlex.split(shlex.quote(nxf_cmd)),
                shell=False,
                cwd=Path.home(),
            )
            self.process.wait()
            # Check if the process was successful
            if self.process.returncode != 0:
                raise RuntimeError

        @thread_worker(
            connect={
                "started": self._pipeline_start,
                "returned": self._pipeline_finish,
                "errored": self._pipeline_fail,
            }
        )
        def _run_pipeline_ssh(nxf_cmd: str, img_paths: list[Path]):
            self.remote_base_dir = str(self.nxf_base_dir).replace(
                self.mounted_path_prefix.text(),
                self.remote_path_prefix.text(),
                1,
            )
            self.mounted_remote_base_dir = str(self.nxf_base_dir)

            remote_img_paths = [
                (
                    Path(
                        str(p).replace(
                            self.mounted_path_prefix.text(),
                            self.remote_path_prefix.text(),
                            1,
                        )
                    )
                )
                for p in img_paths
            ]

            self.store_img_paths(img_paths=remote_img_paths)

            with open(nxf_params_fpath, "r") as f:
                nxf_params = yaml.safe_load(f)

            for k, v in nxf_params.items():
                if isinstance(v, str) and v.startswith(
                    self.mounted_remote_base_dir
                ):
                    nxf_params[k] = v.replace(
                        self.mounted_remote_base_dir,
                        self.remote_base_dir,
                        1,
                    )

            with open(nxf_params_fpath, "w") as f:
                yaml.dump(nxf_params, f)

            # Update Nextflow command to use remote paths for params-file and log
            remote_params_fpath = Path(
                str(nxf_params_fpath).replace(
                    self.mounted_remote_base_dir, self.remote_base_dir, 1
                )
            )

            remote_log_fpath = Path(self.remote_base_dir) / "nextflow.log"
            # Replace local params-file and log with remote ones in the command
            nxf_cmd = nxf_cmd.replace(
                f"-params-file {nxf_params_fpath}",
                f"-params-file {remote_params_fpath}",
                1,
            )
            # Replace local work directory with remote work directory in the command
            remote_work_dir = Path(self.remote_base_dir) / "work"
            nxf_cmd = nxf_cmd.replace(
                f"-w {self.nxf_work_dir}", f"-w {remote_work_dir}", 1
            )
            nxf_cmd = nxf_cmd.replace(
                f"-log '{str(self.nxf_base_dir / 'nextflow.log')}'",
                f"-log '{remote_log_fpath}'",
                1,
            )
            nxf_cmd = self.command_prepend.text() + " && " + nxf_cmd
            show_info(f"Commmand sent via ssh: {nxf_cmd}")

            self._run_command(nxf_cmd)

        # Run the pipeline
        if self.ssh_box.isChecked():
            _run_pipeline_ssh(nxf_cmd, img_paths)
        else:
            _run_pipeline(nxf_cmd)
        self.config_ready.emit()
        self.nxf_params = nxf_params

    def _reset_btns(self):
        """
        Resets the buttons to their original state.
        """
        self.nxf_run_btn.setText("Run Pipeline!")
        self.nxf_run_btn.setEnabled(True)
        self._remove_cancel_btn()

    def _pipeline_start(self):
        # Add a notification that the pipeline has started
        show_info("Pipeline started!")
        # Modify buttons during run
        # Disable run button to avoid issues
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

    def cancel_pipeline(self):
        # Trigger Nextflow to cancel the pipeline
        self.process.send_signal(subprocess.signal.SIGTERM)
        # Reset the progress bar
        self.reset_progress_bar()
        # Remove mask layers that were added
        self.parent.remove_mask_layers()

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
        H, W, num_slices, channels = get_img_dims(layers[0], verbose=False)
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

    def on_click_inspect_cache(self):
        """
        Open the cache directory in the file explorer for a user to inspect, if they want.

        Doesn't do anything else, just opens the directory.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setDirectory(str(self.nxf_base_dir))
        dialog.exec()

    def on_click_clear_cache(self):
        """
        Confirm with the user before clearing the cache.

        Note that the order/location of the buttons depends on OS.
        """
        # Prompt the user to confirm deletion
        prompt_window = QMessageBox()
        prompt_window.setIcon(QMessageBox.Question)
        prompt_window.setText("Are you sure you want to clear the cache?")
        prompt_window.setInformativeText(
            "This will remove all models and results from the cache."
        )
        # Get details about key files
        mask_dirs = [
            i
            for i in (self.nxf_base_dir / "aiod_cache").rglob("*")
            if i.is_dir() and i.name.endswith("_masks")
        ]
        # Count the number of masks
        num_masks = sum(
            len(list(mask_dir.glob("*.rle"))) for mask_dir in mask_dirs
        )
        # Count number of configs
        num_configs = len(
            list((self.nxf_base_dir / "aiod_cache").glob("nxf_params_*.yml"))
        )
        # Count number of checkpoints
        chkpt_dirs = [
            i
            for i in (self.nxf_base_dir / "aiod_cache").rglob("*")
            if i.is_dir() and i.name == "checkpoints"
        ]
        num_chkpts = sum(
            len(list(chkpt_dir.glob("*"))) for chkpt_dir in chkpt_dirs
        )
        # Create message for detailed text
        msg = (
            f"Your cache ({self.nxf_base_dir}) contains the following files:\n"
            + "\n".join(
                [
                    f"{num_masks} masks",
                    f"{num_chkpts} model checkpoints (or related files)",
                    f"{num_configs} Nextflow parameter files",
                ]
            )
        )

        prompt_window.setDetailedText(msg)
        prompt_window.setWindowTitle("Clear cache")
        # Create different buttons for different levels of deletion
        clear_models = prompt_window.addButton(
            "Clear models only", QMessageBox.ButtonRole.ActionRole
        )
        clear_masks = prompt_window.addButton(
            "Clear masks only", QMessageBox.ButtonRole.ActionRole
        )
        clear_all = prompt_window.addButton(
            "Clear all", QMessageBox.ButtonRole.ActionRole
        )
        cancel = prompt_window.addButton(QMessageBox.StandardButton.Cancel)
        prompt_window.setDefaultButton(cancel)
        retval = prompt_window.exec()
        # Check which button was pressed
        clicked_btn = prompt_window.clickedButton()
        if (
            clicked_btn == QMessageBox.StandardButton.Close
            or clicked_btn == cancel
        ):
            return
        elif clicked_btn == clear_models:
            # Delete all 'checkpoints' folders
            for chkpt_dir in chkpt_dirs:
                shutil.rmtree(chkpt_dir)
        elif clicked_btn == clear_masks:
            # Delete all mask subdirectories
            for mask_dir in mask_dirs:
                shutil.rmtree(mask_dir)
        elif clicked_btn == clear_all:
            # Delete the cache directory and all its contents
            shutil.rmtree(self.nxf_base_dir)
            # Reset the base directory
            self.setup_nxf_dir_cmd(base_dir=self.nxf_base_dir)

    def _show_ssh_info(self):
        QMessageBox.information(
            self,
            "SSH Key Help",
            (
                "You need to select your ssh key (looks something like: id_<encryption_algorithm>) in .ssh from your home directory.\n\n"
                "If you can't see hidden folders you may need to toggle visibility:\n"
                "macOS: Command + Shift + .\n"
                "Linux: Ctrl + H (may differ by distro)\n"
                "Windows: Enable 'Hidden items' in the View menu."
            ),
        )

    def _locate_ssh_key(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Key", str(Path.home() / ".ssh")
        )
        if file_path:
            self.ssh_key_path = file_path
            self.ssh_key_label.setText(f"SSH Key: {file_path}")

    def _run_ssh(
        self,
        command: str,
        hostname: str,
        username: str,
        passphrase: str,
        ssh_key_path: str = None,
        target_node: str = None,
    ):
        if not passphrase:
            passphrase = None
        jump = None
        target = None
        try:
            # Connect to the jump host
            jump = paramiko.SSHClient()
            jump.load_system_host_keys()
            jump.set_missing_host_key_policy(paramiko.RejectPolicy())
            jump.connect(
                hostname=hostname,
                username=username,
                key_filename=ssh_key_path,
                passphrase=passphrase,
            )
            if target_node:
                # Get a direct-tcpip channel to the target node through the jump host
                jump_transport = jump.get_transport()
                dest_addr = (target_node, 22)
                local_addr = ("127.0.0.1", 0)
                channel = jump_transport.open_channel(
                    "direct-tcpip", dest_addr, local_addr
                )

                # Connect to the target node using the channel as a proxy
                target = paramiko.SSHClient()
                target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                target.connect(
                    target_node,
                    username=username,
                    key_filename=ssh_key_path,
                    sock=channel,
                    passphrase=passphrase,
                )
            else:
                target = jump
            # Execute command on the target node
            _, stdout, stderr = target.exec_command(command)

            while True:
                stdout_line = stdout.readline()
                stderr_line = stderr.readline()
                if not stdout_line and not stderr_line:
                    break
                if stdout_line:
                    print(stdout_line, end="")
                if stderr_line:
                    print(stderr_line, end="")

        except Exception as e:
            raise Exception(f"Error executing command via jump host: {str(e)}")
        finally:
            if target:
                target.close()
            if jump:
                jump.close()

    def _run_command(self, command):
        hostname = self.hostname.text()
        target_node = self.target_node.text()
        username = self.username.text()
        passphrase = self.passphrase_input.text()
        self._run_ssh(
            command,
            hostname,
            username,
            passphrase,
            self.ssh_key_path,
            target_node,
        )
