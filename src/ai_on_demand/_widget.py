from collections import Counter
from pathlib import Path
import subprocess
import time
import yaml

from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QRadioButton,
    QGroupBox,
    QComboBox,
)
from qtpy.QtGui import QPixmap
import qtpy.QtCore
import numpy as np
import skimage.io

import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

from .model_params import MODEL_PARAMS, MODEL_VERSIONS

# Available models, with displayable names
MODELS = {"unet": "UNet", "sam": "Segment Anything"}
# Specify which models are available for each task
MODEL_DICT = {
    "mito": [k for k in MODELS if k != "sam"],
    "er": [k for k in MODELS if k != "sam"],
    "ne": [k for k in MODELS if k != "sam"],
    "everything": ["sam"],
}


class AIOnDemand(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Handy attributes to check things
        self.selected_task = None
        self.selected_model = None

        # Set the path to watch for masks
        self.mask_path = (
            Path(__file__).parent / "nextflow/modules/models" / "sam_masks"
        )

        # Set overall layout for widget
        self.setLayout(QVBoxLayout())

        # Add Crick logo at top for flavour
        self.logo_label = QLabel()
        logo = QPixmap(
            str(
                Path(__file__).parent
                / "resources"
                / "CRICK_Brandmark_01_transparent.png"
            )
        ).scaledToHeight(150)
        self.logo_label.setPixmap(logo)
        self.logo_label.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.layout().addWidget(self.logo_label)

        # Create radio buttons for selecting task (i.e. organelle)
        self.create_organelle_box()

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.create_model_box()

        # Create the box for selecting the directory, showing img count etc.
        self.create_dir_box()

        # Add the button for running the Nextflow pipeline
        self.create_nxf_button()

    def create_organelle_box(self):
        # Define the box and layout
        self.task_group = QGroupBox("Select organelle to segment:")
        self.task_layout = QVBoxLayout()
        # Define and set the buttons for the different tasks
        # With callbacks to change other options accoridngly
        self.task_buttons = {}
        for name, label in zip(
            ["mito", "er", "ne", "everything"],
            [
                "Mitochondria",
                "Endoplasmic Reticulum",
                "Nuclear Envelope",
                "Everything!",
            ],
        ):
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            btn.clicked.connect(self.update_model_box)
            self.task_layout.addWidget(btn)
            self.task_buttons[name] = btn
        # Add the buttons under the overall group box
        self.task_group.setLayout(self.task_layout)
        # Add to main widget
        self.layout().addWidget(self.task_group)

    def create_model_box(self):
        self.model_group = QGroupBox("Select model to run:")
        self.model_layout = QVBoxLayout()
        # Define and set the buttons for each model
        self.model_buttons = {}
        for name, label in MODELS.items():
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            # Necessary to allow for unchecking, without which they cannot be unselected
            # when switching tasks
            btn.setAutoExclusive(False)
            # Update model parameters for selected model if checked
            btn.clicked.connect(partial(self.update_model_params, name))
            self.model_layout.addWidget(btn)
            self.model_buttons[name] = btn
        # Create a container for the model parameters and containing widget
        # NOTE: Should be light on memory, but this does store for every model
        self.model_param_dict = {}
        self.model_param_widgets = {}
        self.model_widget = None
        # Add a collapsible box for model parameters
        self.model_param_box = QCollapsible("Model parameters:")
        self.model_param_box.setDuration(0)
        self.model_param_box.layout().setContentsMargins(0, 0, 0, 0)
        self.model_param_box.layout().setSpacing(2)
        # Add a tooltip to the button
        self.model_param_box._toggle_btn.setToolTip(
            "Show/hide model parameters if you wish to modify the defaults."
        )
        # Add an initial label to show that no model has been selected
        self.model_param_init_label = QLabel("No model selected.")
        # Add this to the collapsible box
        self.model_param_box.addWidget(self.model_param_init_label)
        # Add the collapsible box to the layout
        self.model_layout.addWidget(self.model_param_box)
        # Finalise layout and adding group to main window for model buttons
        self.model_group.setLayout(self.model_layout)
        self.layout().addWidget(self.model_group)

    def update_model_params(self, model_name):
        # TODO: Add buttons here to switch between modifying parameters and loading yaml
        if self.model_param_box.isExpanded():
            self.model_param_box.collapse()
        # Clear anything present, ignore all errors
        try:
            self.model_param_box.removeWidget(self.model_param_init_label)
            self.model_param_box.removeWidget(self.model_widget)
        except:
            pass
        # Extract the default parameters
        try:
            param_dict = MODEL_PARAMS[model_name]
        except KeyError as e:
            raise e("Default model parameters not found!")
        # Retrieve the widget for this model if already created
        if model_name in self.model_param_widgets:
            self.model_widget = self.model_param_widgets[model_name]
        # Otherwise construct it
        else:
            self.model_widget = self._create_model_param_widget(
                model_name, param_dict
            )
            self.model_param_widgets[model_name] = self.model_widget
        # Set the collapsible box to contain the params for this model
        self.model_param_box.addWidget(self.model_widget)
        self.model_param_box._toggle_btn.setText(
            f"{MODELS[model_name]} parameters:"
        )

    def _create_model_param_widget(self, model_name, param_dict):
        # Create a widget for the model parameters
        model_widget = QWidget()
        model_layout = QGridLayout()
        # Create container for model parameters
        self.model_param_dict[model_name] = {}
        # Add dropdown of model checkpoints/types if available
        self._list_model_checkpoints(model_name, model_layout)
        # Initialize row counter in grid layout
        i = 1
        # Add the default model parameters
        for label, model_param in param_dict.items():
            # Create labels for each of the model parameters
            param_label = QLabel(f"{label}:")
            param_label.setToolTip(model_param.tooltip)
            model_layout.addWidget(param_label, i, 0)
            # Add the default model parameter
            param_value = QLineEdit()
            param_value.setText(str(model_param.default))
            model_layout.addWidget(param_value, i, 1)
            # Store for later retrieval when saving the config
            self.model_param_dict[model_name][label] = {
                "label": param_label,
                "value": param_value,
            }
            i += 1
        # Tighten up margins and set layout
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_widget.setLayout(model_layout)
        return model_widget

    def _list_model_checkpoints(self, model_name, model_layout):
        model_types = MODEL_VERSIONS[model_name]
        # If no types available, just add a default
        if model_types is None:
            model_types = ["default"]
        # Create a label for the dropdown
        param_label = QLabel("Model type:")
        param_label.setToolTip(
            "Select the model type to use. If only 'default' is available, then no model variants are available."
        )
        # Create the dropdown box
        model_type_box = QComboBox()
        model_type_box.addItems(model_types)
        model_type_box.setCurrentIndex(0)
        # Insert into dict for later retrieval
        self.model_param_dict[model_name]["model_type"] = {
            "label": param_label,
            "value": model_type_box,
        }
        # Add to layout
        model_layout.addWidget(param_label, 0, 0)
        model_layout.addWidget(model_type_box, 0, 1)

    def update_model_box(self):
        """The model box updates according to what's defined for each task"""
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
                self.selected_task = task_name
                # Get the models available for this task
                avail_models = MODEL_DICT[task_name]
                # Disable selection of all models not selected
                for model_name, model_btn in self.model_buttons.items():
                    if model_name not in avail_models:
                        # Grey out and disable ineligible options
                        model_btn.setEnabled(False)
                        model_btn.setStyleSheet("color: gray")
                        # Uncheck if already checked and no longer available
                        model_btn.setChecked(False)
                    else:
                        model_btn.setEnabled(True)
                        model_btn.setStyleSheet("")

    def _check_models(self):
        for model_name, model_btn in self.model_buttons.items():
            if model_btn.isChecked():
                self.selected_model = model_name
                return

    def create_dir_box(self):
        # TODO: Simultaneously allow for drag+dropping, probably a common use pattern
        self.dir_group = QGroupBox("Data selection:")
        self.dir_layout = QVBoxLayout()
        # Add an output to show the selected path
        self.images_dir_label = QLabel("Image folder: not selected.")
        self.images_dir_label.setWordWrap(True)
        # Create empty container for selected image filepaths
        self.all_img_files = None
        # Create empty container for the image directory
        self.images_dir = None
        # Create empty counter to show image load progress
        self.load_img_counter = 0
        # Create a button to navigate to a directory to take images from
        self.dir_btn = QPushButton("Select directory")
        self.dir_btn.clicked.connect(self.browse_directory)
        self.dir_btn.setToolTip(
            "Select folder/directory of images to use as input to the model."
        )
        self.dir_layout.addWidget(self.dir_btn)
        # Add an output to show the counts
        self.img_counts = QLabel()
        self.img_counts.setWordWrap(True)
        self.img_counts.setText("No files selected.")
        self.dir_layout.addWidget(self.img_counts)
        # self.images_dir_label = QLineEdit("")  # TODO: Make it editable fo ruser input too
        self.dir_layout.addWidget(self.images_dir_label)
        # Add a button for viewing the images within napari
        # Optional as potentially taxing, and not necessary
        self.view_img_btn = QPushButton("View selected images")
        self.view_img_btn.setToolTip(
            "Load selected images into napari to view."
        )
        self.view_img_btn.clicked.connect(self.view_images)
        self.dir_layout.addWidget(self.view_img_btn)
        # Sort out layout and add to main widget
        self.dir_group.setLayout(self.dir_layout)
        self.layout().addWidget(self.dir_group)

    def abspath(self, root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    def browse_directory(self):
        result = QFileDialog.getExistingDirectory(
            self, caption="Select image directory", directory=None
        )
        # If a new directory is selected, reset the load button text
        if result != self.images_dir:
            self.view_img_btn.setText("View selected images")
            self.view_img_btn.setEnabled(True)
        if result != "":
            self.images_dir = result
            self.images_dir_label.setText(f"Image folder:\n{self.images_dir}")
            self._count_files()

    def type_directory(self):
        """Allow for user to type a directory?

        Will require checking that the path is valid, and using the ToolTip if it's not
        (or a similar pop-up/warning object) to raise the issue.
        """
        return NotImplementedError

    def _count_files(self):
        """Function to extract all the files in a given path,
        and return a count (broken down by extension)
        """
        txt = ""
        # Get all the files in the given path
        self.all_img_files = list(Path(self.images_dir).glob("*"))
        # Get all the extensions in the path
        extension_counts = Counter([i.suffix for i in self.all_img_files])
        # Sort by highest and get the suffixes and their counts
        ext_counts = extension_counts.most_common()
        if len(ext_counts) > 1:
            # Nicely format the list of files and their extensions
            for i, (ext, count) in enumerate(ext_counts):
                if i == (len(ext_counts) - 1):
                    txt += f"and {count} {ext}"
                else:
                    txt += f"{count} {ext}, "
        else:
            txt += f"{ext_counts[0][1]} {ext_counts[0][0]}"
        txt += " files selected."
        self.img_counts.setText(txt)

    def view_images(self):
        self.view_img_btn.setEnabled(False)
        # Return if there's nothing to show
        if self.all_img_files is None:
            return
        # Reset counter
        self.load_img_counter = 0

        # Create separate thread worker to avoid blocking
        @thread_worker(
            connect={
                "returned": self._add_image,
                "finished": self._reset_view_btn,
            }
        )
        def _load_image(fpath):
            return skimage.io.imread(fpath), fpath

        # Load each image in a separate thread
        for fpath in self.all_img_files:
            _load_image(fpath)
        # NOTE: This does not work well for a directory of large images on a remote directory
        # But that would trigger loading GBs into memory over network, which is...undesirable
        self.view_img_btn.setText("Loading...")

    def _add_image(self, res):
        img, fpath = res
        self.viewer.add_image(img, name=fpath.name)
        self.load_img_counter += 1
        self.view_img_btn.setText(
            f"Loading...({self.load_img_counter}/{len(self.all_img_files)} images loaded)."
        )
        if self.load_img_counter == len(self.all_img_files):
            self.view_img_btn.setText(
                f"All ({self.load_img_counter}/{len(self.all_img_files)}) images loaded."
            )

    def _reset_view_btn(self):
        self.view_img_btn.setEnabled(True)

    def watch_mask_files(self):
        # Wait for at least one image to load as layers if not present
        if not self.viewer.layers:
            time.sleep(0.5)
        # Create the Labels layers for each image
        for fpath in self.all_img_files:
            # If images still not loaded, add dummy array
            try:
                img_shape = self.viewer.layers[f"{fpath.name}"].data.shape
            except KeyError:
                img_shape = (1000, 1000)
            # Set the name following convention
            name = f"{fpath.stem}_masks"
            # Add a Labels layer for this file
            self.viewer.add_labels(
                np.zeros(img_shape, dtype=int), name=name, visible=False
            )

        # NOTE: Wrapper as self/class not available at runtime
        @thread_worker(connect={"yielded": self.update_masks})
        def _watch_mask_files(self):
            # Enable the watcher
            self.watcher_enabled = True
            # Initialize empty container for storing mask filepaths
            self.mask_fpaths = []
            # Loop and yield any changes infinitely while enabled
            while self.watcher_enabled:
                # Get all files
                current_files = list(self.mask_path.glob("*.npy"))
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
                # Get how many mask files there should be
                num_images = len(self.all_img_files)
                # If all images have complete masks, deactivate watcher
                if all(masks_finished) and (len(masks_finished) == num_images):
                    print("Deactivating watcher...")
                    self.watcher_enabled = False

        # Call the nested function
        _watch_mask_files(self)

    def update_masks(self, new_files):
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            mask_arr = np.load(f)
            # Extract the relevant Labels layer
            split_names = f.stem.split("_masks")
            layer_name = split_names[0] + "_masks"
            label_layer = self.viewer.layers[layer_name]
            # label_layer.num_colours = mask_arr.max()+1
            # Insert mask data
            label_layer.data = mask_arr
            label_layer.visible = True
            slice_num = split_names[-1].replace("_", "")
            # Switch viewer to latest slice
            if slice_num == "all":
                slice_num = label_layer.data.shape[0]
            else:
                slice_num = int(slice_num)
            self.viewer.dims.set_point(0, slice_num)

    def create_nxf_button(self):
        self.nxf_group = QGroupBox("Nextflow Pipeline:")
        self.nxf_layout1 = QVBoxLayout()
        self.nxf_layout2 = QHBoxLayout()
        # Create a drop-down box to select the execution profile
        self.nxf_profile_label = QLabel("Execution profile:")
        self.nxf_profile_label.setToolTip("Select the execution profile to use.")
        self.nxf_profile_box = QComboBox()
        # Get the available profiles from config dir
        config_dir = Path(__file__).parent / "nextflow" / "profiles"
        avail_confs = [str(i.stem) for i in config_dir.glob("*.conf")]
        self.nxf_profile_box.addItems(avail_confs)
        self.nxf_layout2.addWidget(self.nxf_profile_label)
        self.nxf_layout2.addWidget(self.nxf_profile_box)
        # Nest this layout within the main layout
        self.nxf_layout1.addLayout(self.nxf_layout2)
        # Create a button to navigate to a directory to take images from
        self.nxf_btn = QPushButton("Run Pipeline!")
        self.nxf_btn.clicked.connect(self.run_pipeline)
        self.nxf_btn.setToolTip(
            "Run the pipeline with the chosen organelle(s), model, and images."
        )
        self.nxf_layout1.addWidget(self.nxf_btn)
        self.nxf_group.setLayout(self.nxf_layout1)
        self.layout().addWidget(self.nxf_group)

    def store_img_paths(self):
        self.img_list_fpath = Path(__file__).parent / "all_img_paths.txt"

        with open(self.img_list_fpath, "w") as output:
            output.write("\n".join([str(i) for i in self.all_img_files]))

    def create_nextflow_params(self):
        config_path, model_type = self.get_model_config()
        nxf_params = {}
        nxf_params["img_dir"] = str(self.img_list_fpath)
        nxf_params["model"] = self.selected_model
        nxf_params["model_config"] = config_path
        nxf_params["model_type"] = model_type
        nxf_params["task"] = self.selected_task
        # TODO: Implement profiles for this to configure SLURM
        nxf_params["executor"] = self.nxf_profile_box.currentText()
        return nxf_params

    def get_model_config(self):
        # Get the current dictionary of widgets for selected model
        model_dict_orig = self.model_param_dict[self.selected_model]
        # Get the relevant default params for this model
        default_params = MODEL_PARAMS[self.selected_model]
        # Reformat the dict to pipe into downstream model run scripts
        model_dict = {}
        # Extract params from model param widgets
        for param_name, sub_dict in model_dict_orig.items():
            if isinstance(sub_dict["value"], QLineEdit):
                param_value = sub_dict["value"].text()
            elif isinstance(sub_dict["value"], QComboBox):
                param_value = sub_dict["value"].itemData(
                    sub_dict["value"].currentIndex()
                )
            else:
                raise NotImplementedError
            # Extract the original/intended dtype and cast what's in the box
            if param_name != "model_type":
                orig_dtype = default_params[param_name].dtype
                model_dict[default_params[param_name].arg] = orig_dtype(param_value)
        # Extract the model type
        model_type = model_dict_orig["model_type"]["value"].currentText()
        # Define save path for the model config
        config_dir = Path(__file__).parent / "nextflow" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        model_config_fpath = (
            config_dir / f"{self.selected_model}-{model_type}_config.yaml"
        )
        # Save the yaml config
        with open(model_config_fpath, "w") as f:
            yaml.dump(model_dict, f)
        return model_config_fpath, model_type

    def run_pipeline(self):
        # Start with some error checking to ensure that everything has been properly specified
        # Check a task/organelle has been selected
        if self.selected_task is None:
            raise ValueError("No task/organelle has been selected!")
        # Check a model has been selected
        self._check_models()
        if self.selected_model is None:
            raise ValueError("No model has been selected!")
        # Reset LayerList
        self.viewer.layers.clear()
        # Check a directory of images has been given
        # NOTE: They do not have to have been loaded, but to show feedback they will be loaded
        self.view_images()
        # Create a text file of the image paths
        self.store_img_paths()
        # Create the params file for Nextflow
        nxf_params = self.create_nextflow_params()
        # Start the mask file watcher
        self.watch_mask_files()

        @thread_worker(
            connect={
                "returned": self._pipeline_finish,
                "errored": self._pipeline_fail,
            }
        )
        def _run_pipeline(self, nxf_params):
            # Update the button to signify it's running
            self.nxf_btn.setText("Running Pipeline...")
            # Disable the button to avoid issues
            self.nxf_btn.setEnabled(False)
            # Get the path the main Nextflow entry pipeline
            nextflow_script = Path(__file__).parent / "nextflow" / "main.nf"
            exec_str = f"nextflow run {str(nextflow_script)}"
            # Add the command line arguments
            for k, v in nxf_params.items():
                exec_str += f" --{k}={v}"
            # Add the execution profile
            exec_str += f" -profile {self.nxf_profile_box.currentText()}"
            # Run the pipeline!
            subprocess.run(exec_str, shell=True)
            # TODO: Have some error-handling/polling

        _run_pipeline(self, nxf_params)

    def _pipeline_finish(self):
        # Add a notification that the pipeline has finished
        show_info("Pipeline finished!")
        # Reset the run pipeline button
        self.nxf_btn.setText("Run Pipeline!")
        self.nxf_btn.setEnabled(True)

    def _pipeline_fail(self, exc):
        # Reset the run pipeline button
        self.nxf_btn.setText("Run Pipeline!")
        self.nxf_btn.setEnabled(True)
        raise exc
