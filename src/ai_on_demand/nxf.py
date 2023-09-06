from pathlib import Path
import subprocess

import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QGroupBox,
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QProgressBar,
)
from ai_on_demand.models import MODEL_TASK_VERSIONS, MODEL_DISPLAYNAMES
from ai_on_demand.utils import sanitise_name


class NxfWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, pipeline: str, parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent = parent

        # Set the layout
        self.setLayout(QGridLayout())

        # Define attributes that may be useful outside of this class
        # or throughout it
        self.nxf_repo = "FrancisCrickInstitute/Segment-Flow"

        self.pipeline = pipeline
        # Available pipelines and their funcs
        self.pipelines = {
            "inference": self.setup_inference,
            "finetuning": self.setup_finetuning,
        }

        # Create the widget itself
        self.widget = QGroupBox("Nextflow Pipeline:")

        # Create a drop-down box to select the execution profile
        self.nxf_profile_label = QLabel("Execution profile:")
        self.nxf_profile_label.setToolTip(
            "Select the execution profile to use."
        )
        self.nxf_profile_box = QComboBox()
        # Get the available profiles from config dir
        config_dir = Path(__file__).parent / "Segment-Flow" / "profiles"
        avail_confs = [str(i.stem) for i in config_dir.glob("*.conf")]
        self.nxf_profile_box.addItems(avail_confs)
        self.layout().addWidget(self.nxf_profile_label, 0, 0)
        self.layout().addWidget(self.nxf_profile_box, 0, 1)
        # Create a button to navigate to a directory to take images from
        self.nxf_run_btn = QPushButton("Run Pipeline!")
        self.nxf_run_btn.clicked.connect(self.run_pipeline)
        self.nxf_run_btn.setToolTip(
            "Run the pipeline with the chosen organelle(s), model, and images."
        )
        self.layout().addWidget(self.nxf_run_btn, 1, 0, 1, 2)

        # Add a button for exporting masks
        self.export_masks_btn = QPushButton("Export masks")
        self.export_masks_btn.clicked.connect(self.on_click_export)
        self.export_masks_btn.setToolTip(
            "Export the segmentation masks to a directory."
        )
        self.export_masks_btn.setEnabled(False)
        # TODO: Add dropdown for different formats to export to
        self.layout().addWidget(self.export_masks_btn, 2, 0, 1, 1)

        self.widget.setLayout(self.layout())

        # If given a parent at creation, add this widget to the parent's layout
        if parent is not None:
            parent.layout().addWidget(self.widget)

    def store_img_paths(self, img_paths):
        """
        Writes the provided image paths to a file to pass into Nextflow.

        TODO: May be subject to complete rewrite with dask/zarr
        """
        self.img_list_fpath = Path(__file__).parent / "all_img_paths.txt"
        # Write the image paths into a newline-separated text file
        with open(self.img_list_fpath, "w") as output:
            output.write("\n".join([str(i) for i in img_paths]))

    def setup_inference(self, nxf_params=None):
        """
        Runs the inference pipeline in Nextflow.

        `nxf_params` is a dict containing everything that Nextflow needs at the command line.

        `parent` is a parent widget, which is expected to contain the necessary info to construct `nxf_params`.

        NOTE: A lot of this will need to be switched when Model subwidget created.
        """
        # nxf_cmd = f"nextflow run {self.nxf_repo} -entry inference"
        # Set the base Nextflow command
        nxf_cmd = f"nextflow run {self.nxf_repo} -r master"
        # nxf_params can only be given when used standalone, which is rare
        if nxf_params is not None:
            return nxf_cmd, nxf_params
        # Construct the Nextflow params if not given
        parent = self.parent
        if parent.model_config is None:
            # TODO: Switch to Model widget
            config_path = parent.get_model_config()
        else:
            config_path = parent.model_config
        # Extract the current model version selected
        selected_model = MODEL_DISPLAYNAMES[
            parent.model_dropdown.currentText()
        ]
        selected_variant = parent.model_version_dropdown.currentText()
        selected_task = parent.selected_task
        # Construct the params to be given to Nextflow
        nxf_params = {}
        nxf_params["img_dir"] = str(self.img_list_fpath)
        nxf_params["model"] = selected_model
        nxf_params["model_config"] = config_path
        nxf_params["model_type"] = sanitise_name(selected_variant)
        nxf_params["task"] = selected_task
        # Extract the model checkpoint location and location type
        checkpoint_info = MODEL_TASK_VERSIONS[selected_model][selected_task][
            selected_variant
        ]
        if "url" in checkpoint_info:
            nxf_params["model_chkpt_type"] = "url"
            nxf_params["model_chkpt_loc"] = checkpoint_info["url"]
            nxf_params["model_chkpt_fname"] = checkpoint_info["filename"]
        elif "dir" in checkpoint_info:
            nxf_params["model_chkpt_type"] = "dir"
            nxf_params["model_chkpt_loc"] = checkpoint_info["dir"]
            nxf_params["model_chkpt_fname"] = checkpoint_info["filename"]
        return nxf_cmd, nxf_params

    def setup_finetuning(self):
        """
        Runs the finetuning pipeline in Nextflow.
        """
        raise NotImplementedError

    def run_pipeline(self):
        # TODO: Add any general steps prior to running the pipeline
        # Store the image paths
        self.image_path_dict = self.parent.image_path_dict
        self.store_img_paths(img_paths=self.image_path_dict.values())
        # Ensure the pipeline is valid
        assert (
            self.pipeline in self.pipelines.keys()
        ), f"Pipeline {self.pipeline} not found!"
        # Get the pipeline-specific stuff
        nxf_cmd, nxf_params = self.pipelines[self.pipeline]()
        # Add the selected profile to the command
        nxf_cmd += f" -profile {self.nxf_profile_box.currentText()}"
        # Add the parameters to the command
        for param, value in nxf_params.items():
            nxf_cmd += f" --{param}={value}"

        self.parent.view_images()
        self.parent.watch_mask_files()

        @thread_worker(
            connect={
                "returned": self._pipeline_finish,
                "errored": self._pipeline_fail,
            }
        )
        def _run_pipeline(nxf_cmd: str):
            # Run the command
            subprocess.run(
                nxf_cmd, shell=True, cwd=Path(__file__).parent, check=True
            )

        # Modify buttons during run
        self.export_masks_btn.setEnabled(False)
        # Disable the button to avoid issues
        # TODO: Enable multiple job execution, may require -bg flag
        self.nxf_run_btn.setEnabled(False)
        # Update the button to signify it's running
        self.nxf_run_btn.setText("Running Pipeline...")
        # Run the pipeline
        _run_pipeline(nxf_cmd)

    def _reset_btns(self):
        """
        Resets the buttons to their original state.
        """
        self.nxf_run_btn.setText("Run Pipeline!")
        self.nxf_run_btn.setEnabled(True)
        self.export_masks_btn.setEnabled(True)

    def _pipeline_finish(self):
        # Add a notification that the pipeline has finished
        show_info("Pipeline finished!")
        self._reset_btns()

    def _pipeline_fail(self, exc):
        show_info("Pipeline failed! See terminal for details")
        print(exc)
        self._reset_btns()

    def create_progress_bars(self):
        print("Making progress bars")
        # Create the overall widget
        self.progress_bar_widget = QGroupBox("Progress Bars:")
        # progress_widget_layout = QVBoxLayout()

        progress_bar_layout = QGridLayout()

        # If only 2D images are present, then max slice for all will be 1
        if self.viewer.dims.ndim == 2:
            max_slice = 1
        # Construct a progress bar for each model
        self.progress_bar_dict = {}
        for row_num, img_name in enumerate(self.image_path_dict):
            # Extract the number of slices
            if self.viewer.dims.ndim > 2:
                try:
                    # Assumes ([C], D, H, W) ordering
                    max_slice = self.viewer.layers[img_name].data.shape[-3]
                # If the image hasn't loaded yet, set to 0 and fill in later
                except KeyError:
                    max_slice = 0
            # Create the pbar and set the range
            pbar = QProgressBar()
            pbar.setRange(0, max_slice)
            pbar.setValue(0)
            # Create the label associated with the progress bar
            pbar_label = QLabel(f"{img_name}:")

            progress_bar_layout.addWidget(pbar_label, row_num, 0)
            progress_bar_layout.addWidget(pbar, row_num, 1)

            self.progress_bar_dict[img_name] = pbar

        # Scroll area
        # scroll_area = QScrollArea()
        # scroll_area.setWidget(self.progress_bar_widget)
        # progress_widget_layout.addWidget(scroll_area)
        # progress_widget_layout.addLayout(progress_bar_layout)
        self.progress_bar_widget.setLayout(progress_bar_layout)

        # self.layout().addWidget(self.progress_bar_widget)

    def update_progress_bars(self):
        raise NotImplementedError

    def on_click_export(self):
        """
        Callback for when the export button is clicked. Opens a dialog to select a directory to save the masks to.
        """
        export_dir = QFileDialog.getExistingDirectory(
            self, caption="Select directory to save masks", directory=None
        )
        # Get the current viewer
        viewer = self.parent.viewer if self.parent is not None else None
        # FIXME: How to handle if parent doesn't exist? Will this ever happen?
        # Get all the mask layers
        mask_layers = []
        # FIXME: self.parent will be replaced when all widgets become modular
        for img_name in self.image_path_dict:
            layer_name = f"{img_name}_masks_{self.parent.selected_model}-{sanitise_name(self.parent.selected_variant)}"
            if layer_name in viewer.layers:
                mask_layers.append(viewer.layers[layer_name])
        # Extract the data from each of the layers, and save the result in the given folder
        for mask_layer in mask_layers:
            np.save(
                Path(export_dir) / f"{mask_layer.name}.npy", mask_layer.data
            )
