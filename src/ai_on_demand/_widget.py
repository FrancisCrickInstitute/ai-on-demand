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
from napari.layers import Image

from .models import MODEL_INFO, MODEL_DISPLAYNAMES, TASK_MODELS
from .tasks import TASK_NAMES


class AIOnDemand(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Connect to the viewer to some callbacks
        self.viewer.layers.events.inserted.connect(self.on_layer_added)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)

        # Handy attributes to check things
        self.selected_task = None
        self.selected_model = None

        # Set the path to watch for masks
        self.mask_path = (
            Path(__file__).parent / "nextflow/modules/models" / "sam_masks"
        )

        # Set selection colour
        self.colour_selected = "#F7AD6F"

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
        self.create_nxf_box()

    def on_layer_added(self, event):
        """
        Triggered whenever there is a new layer added to the viewer.

        Checks if the layer is an image, and if so, adds it to the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = event.value.source.path
            # Insert into the overall dict of images and their paths (if path is present)
            # This will be None when we are viewing arrays loaded separately from napari
            if img_path is not None:
                self.image_path_dict[Path(img_path).name] = Path(img_path)
            # Then update the counts of files (and their types) with the extra image
            self.update_file_count()

    def on_layer_removed(self, event):
        """
        Triggered whenever a layer is removed from the viewer.

        Checks if the layer is an image, and if so, removes it from the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = Path(event.value.source.path)
            # Remove from the list of images
            if img_path.name in self.image_path_dict:
                del self.image_path_dict[img_path.name]
            # Update file count with image removed
            self.update_file_count()

    def create_organelle_box(self):
        """
        Create the box for selecting the task (i.e. organelle) to segment.
        """
        # Define the box and layout
        self.task_group = QGroupBox("Select organelle to segment:")
        self.task_layout = QVBoxLayout()
        # Define and set the buttons for the different tasks
        # With callbacks to change other options accoridngly
        self.task_buttons = {}
        for name, label in TASK_NAMES.items():
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            btn.clicked.connect(self.on_click_task)
            self.task_layout.addWidget(btn)
            self.task_buttons[name] = btn
        # Add the buttons under the overall group box
        self.task_group.setLayout(self.task_layout)
        # Add to main widget
        self.layout().addWidget(self.task_group)

    def on_click_task(self):
        """
        Callback for when a task button is clicked.

        Updates the model box to show only the models available for the selected task.
        """
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
                self.selected_task = task_name
        # Collapse the modify params or config widgets if open
        # if self.model_param_btn.isChecked():
        #     self.model_param_btn.click()
        # Update the model box for the selected task
        self.update_model_box(self.selected_task)

    def create_model_box(self):
        """
        Create the box for selecting the model (and model version) to run.

        Also contains the widgets for modifying the model parameters and/or loading a config file.
        """
        self.model_group = QGroupBox("Model:")
        self.model_layout = QVBoxLayout()

        model_box_layout = QGridLayout()
        # Create a label for the dropdown
        model_label = QLabel("Select model:")
        # Dropdown of available models
        self.model_dropdown = QComboBox()
        self.model_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        # Store initial message to handle erroneous clicking
        self.model_name_init = "Select a task first!"
        self.model_dropdown.addItems([self.model_name_init])
        # Connect function when new model selected
        self.model_dropdown.activated.connect(self.on_model_select)
        model_box_layout.addWidget(model_label, 0, 0)
        model_box_layout.addWidget(self.model_dropdown, 0, 1, 1, 2)
        # Add label + dropdown for model variants/versions
        model_version_label = QLabel("Select version:")
        model_version_label.setToolTip(
            "Select the model version to use."
            "Versions can vary either by intended functionality, or are specifically for reproducibility."
        )
        self.model_version_dropdown = QComboBox()
        self.model_version_dropdown.setSizeAdjustPolicy(
            QComboBox.AdjustToContents
        )
        self.model_version_dropdown.addItems(["Select a model first!"])
        model_box_layout.addWidget(model_version_label, 1, 0)
        model_box_layout.addWidget(self.model_version_dropdown, 1, 1, 1, 2)

        self.model_layout.addLayout(model_box_layout)

        # Store model config location if given
        self.model_config = None

        # Create container for switching between setting params and loading config
        self.params_config_widget = QWidget()
        self.params_config_layout = QHBoxLayout()
        # Create button for displaying model param options
        self.model_param_btn = QPushButton("Modify Params")
        self.model_param_btn.setToolTip(
            "Open options for modifying model parameters directly in napari."
        )
        self.model_param_btn.setCheckable(True)
        self.model_param_btn.setStyleSheet(
            f"QPushButton:checked {{background-color: {self.colour_selected}}}"
        )
        self.model_param_btn.clicked.connect(self.on_click_model_params)
        self.params_config_layout.addWidget(self.model_param_btn)
        # Create button for displaying model config options
        self.model_config_btn = QPushButton("Load Config")
        self.model_config_btn.setToolTip(
            "Open options for loading a config file to pass to the model."
        )
        self.model_config_btn.setCheckable(True)
        self.model_config_btn.setStyleSheet(
            f"QPushButton:checked {{background-color: {self.colour_selected}}}"
        )
        self.model_config_btn.clicked.connect(self.on_click_model_config)
        self.params_config_layout.addWidget(self.model_config_btn)
        # Reduce unnecessary margins/spacing
        self.params_config_layout.setContentsMargins(0, 0, 0, 0)
        self.params_config_layout.setSpacing(5)
        self.params_config_widget.setLayout(self.params_config_layout)
        # Add the widgets to the overall container
        self.model_layout.addWidget(self.params_config_widget)

        # Create widgets for the two options
        self.create_model_param_widget()
        self.create_model_config_widget()

        # Finalise layout and adding group to main window for model buttons
        self.model_group.setLayout(self.model_layout)
        self.layout().addWidget(self.model_group)

    def on_model_select(self):
        """
        Callback for when a model button is clicked.

        Updates model params & config widgets for selected model.
        """
        # Extract selected model
        model_name = self.model_dropdown.currentText()
        if model_name == self.model_name_init:
            return
        self.selected_model = MODEL_DISPLAYNAMES[model_name]
        # Update the dropdown for the model variants
        self.model_version_dropdown.clear()
        model_versions = MODEL_INFO[self.selected_model]["versions"]
        if model_versions is None:
            model_versions = ["default"]
        self.model_version_dropdown.addItems(model_versions)
        # Update the model params & config widgets for the selected model
        self.update_model_param_config(self.selected_model)
        # TODO: Clear the model config file

    def on_click_model_params(self):
        # Uncheck config button
        if self.model_config_btn.isChecked():
            self.model_config_btn.setChecked(False)
            self.model_config_widget.setVisible(False)
        # Change visibility depending on checked status
        if self.model_param_btn.isChecked():
            self.curr_model_param_widget.setVisible(True)
            self.model_param_widget.setVisible(True)
        else:
            self.curr_model_param_widget.setVisible(False)
            self.model_param_widget.setVisible(False)

    def on_click_model_config(self):
        # Uncheck param button
        if self.model_param_btn.isChecked():
            self.model_param_btn.setChecked(False)
            self.curr_model_param_widget.setVisible(False)
            self.model_param_widget.setVisible(False)
        # Change visibility depending on checked status
        if self.model_config_btn.isChecked():
            self.model_config_widget.setVisible(True)
        else:
            self.model_config_widget.setVisible(False)

    def create_model_config_widget(self):
        """
        Creates the widget for loading a model config file.
        """
        self.model_config_widget = QWidget()
        self.model_config_layout = QVBoxLayout()

        # Add the button for loading a config file
        self.model_config_load_btn = QPushButton("Select model config file")
        self.model_config_load_btn.clicked.connect(self.select_model_config)
        self.model_config_load_btn.setToolTip(
            "Select a config file to be used for the selected model."
            "Note that no checking/validation is done on the config file, it is just given to the model."
        )
        self.model_config_layout.addWidget(self.model_config_load_btn)
        # Add a label to display the selected config file (if any)
        self.model_config_label = QLabel("No model config file selected.")
        self.model_config_label.setWordWrap(True)
        self.model_config_layout.addWidget(self.model_config_label)
        # Set the overall widget layout
        self.model_config_widget.setLayout(self.model_config_layout)

        self.model_config_widget.setVisible(False)
        self.model_layout.addWidget(self.model_config_widget)

    def create_model_param_widget(self):
        """
        Creates the widget for modifying model parameters.
        """
        self.model_param_widget = QWidget()
        self.model_param_layout = QVBoxLayout()
        # Create a container for the model parameters and containing widget
        # NOTE: Should be light on memory, but this does store for every model!
        self.model_param_dict = {}
        self.model_param_widgets_dict = {}
        # Create initial widget as no model has been selected
        init_widget = QWidget()
        init_widget.setLayout(QVBoxLayout())
        # Add an initial label to show that no model has been selected
        model_param_init_label = QLabel("No model selected.")
        # Add this to the collapsible box
        init_widget.layout().addWidget(model_param_init_label)
        self.curr_model_param_widget = init_widget
        # Store initial widget so we can revert state if model deselected
        self.model_param_widgets_dict["init"] = init_widget
        # Add the initial widget to the main model param widget
        self.model_param_layout.addWidget(init_widget)
        self.model_param_widget.setLayout(self.model_param_layout)
        # Disable showing widget until selected to view
        self.model_param_widget.setVisible(False)
        self.model_layout.addWidget(self.model_param_widget)

    def update_model_param_config(self, model_name):
        """
        Updates the model param and config widgets for a specific model.

        Currently only updates the model widget as the config is now constant.
        """
        # Update the model parameters
        self.update_model_param_widget(model_name)

    def update_model_param_widget(self, model_name):
        """
        Updates the model param widget for a specific model
        """
        # Skip if initial message is still showing and clicked
        if model_name not in MODEL_INFO:
            return
        # Remove the current model param widget
        self.model_param_layout.removeWidget(self.curr_model_param_widget)
        self.curr_model_param_widget.setParent(None)
        # Extract the default parameters
        try:
            param_dict = MODEL_INFO[model_name]["params"]
        except KeyError as e:
            raise e("Default model parameters not found!")
        # Retrieve the widget for this model if already created
        if model_name in self.model_param_widgets_dict:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                model_name
            ]
        # Otherwise construct it
        else:
            self.curr_model_param_widget = self._create_model_params_widget(
                model_name, param_dict
            )
            self.model_param_widgets_dict[
                model_name
            ] = self.curr_model_param_widget
        # Set the collapsible box to contain the params for this model
        self.model_param_layout.addWidget(self.curr_model_param_widget)
        # Ensure it's visible if the params button is pressed
        if self.model_param_btn.isChecked():
            self.curr_model_param_widget.setVisible(True)
        else:
            self.curr_model_param_widget.setVisible(False)

    def _create_model_params_widget(self, model_name, param_dict):
        """
        Creates the widget for a specific model's parameters to swap in and out
        """
        # Create a widget for the model parameters
        model_widget = QWidget()
        model_layout = QGridLayout()
        # Create container for model parameters
        self.model_param_dict[model_name] = {}
        # Add the default model parameters
        for i, (label, model_param) in enumerate(param_dict.items()):
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
        # Tighten up margins and set layout
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_widget.setLayout(model_layout)
        return model_widget

    def update_model_box(self, task_name):
        """The model box updates according to what's defined for each task."""
        # Clear and set available models in dropdown
        self.model_dropdown.clear()
        self.model_dropdown.addItems(
            [
                MODEL_INFO[model]["display_name"]
                for model in TASK_MODELS[task_name]
            ]
        )
        # Technically the first model in the list is now selected
        # So update everything accordingly
        self.on_model_select()

    def select_model_config(self):
        """
        Opens a file dialog for selecting the model config.

        Filters can be applied to restrict selection, which may help if restricting processing of those files.
        """
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select a model config",
            str(Path.home()),
            "",
        )
        #
        if fname == "":
            self.model_config_label.setText("No model config file selected.")
            self.model_config = None
            return
        fname = Path(fname)
        # Add a label to show that the config was selected
        self.model_config_label.setText(
            f"Model config file ({fname.name}) selected."
        )
        # Register config location for use in the pipeline
        self.model_config = fname

    def create_dir_box(self):
        """
        Create the box for selecting the directory to take images from, and optionally view them.
        
        Displays the number and types of files found in the selected directory.
        """
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
        # Create container for image paths
        self.image_path_dict = {}
        # Do a quick check to see if the user has added any images already
        if self.viewer.layers:
            for img_layer in self.viewer.layers:
                if isinstance(img_layer, Image):
                    try:
                        img_path = Path(img_layer.source.path)
                        self.image_path_dict[img_path.name] = img_path
                    except AttributeError:
                        continue
        # Create a button to select individual images from
        self.img_btn = QPushButton("Select image files")
        self.img_btn.clicked.connect(self.browse_imgs_files)
        self.img_btn.setToolTip(
            "Select individual image files to use as input to the model."
        )
        self.dir_layout.addWidget(self.img_btn, 0, 0)
        # TODO: What happens if multiple directories are selected? Is this possible?
        # Create a button to navigate to a directory to take images from
        self.dir_btn = QPushButton("Select directory")
        self.dir_btn.clicked.connect(self.browse_directory_imgs)
        self.dir_btn.setToolTip(
            "Select folder/directory of images to use as input to the model."
        )
        self.dir_layout.addWidget(self.dir_btn)
        # Add an output to show the counts
        self.img_counts = QLabel("No files selected.")
        self.img_counts.setWordWrap(True)
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

    def browse_directory_imgs(self):
        """
        Opens a dialog for selecting a directory that contains images to segment.
        """
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

        Will require checking that the path is valid, and using the ToolTip if
        it's not (or a similar pop-up/warning object) to raise the issue.
        """
        return NotImplementedError

    def update_file_count(self, paths=None):
        """
        Function to extract all the files in a given path, and return a count
        (broken down by extension)
        """
        # Reinitialise text
        txt = ""
        # Add paths to the overall list if specific ones need adding
        if paths is not None:
            for img_path in paths:
                img_path = Path(img_path)
                self.image_path_dict[img_path.name] = img_path
        # If no files remaining, reset message and return
        if len(self.image_path_dict) == 0:
            self.img_counts.setText(self.init_file_msg)
            return
        # Get all the extensions in the path
        extension_counts = Counter(
            [i.suffix for i in self.image_path_dict.values()]
        )
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
        """
        Loads the selected images into napari for viewing (in separate threads).
        """
        # Return if there's nothing to show
        if len(self.image_path_dict) == 0:
            return
        # Check if there are images to load that haven't been already
        viewer_imgs = [
            Path(i._source.path)
            for i in self.viewer.layers
            if isinstance(i, Image)
        ]
        imgs_to_load = [
            i for i in self.image_path_dict.values() if i not in viewer_imgs
        ]
        if imgs_to_load == []:
            return
        self.view_img_btn.setEnabled(False)
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
        for fpath in imgs_to_load:
            _load_image(fpath)
        # NOTE: This does not work well for a directory of large images on a remote directory
        # But that would trigger loading GBs into memory over network, which is...undesirable
        self.view_img_btn.setText("Loading...")

    def _add_image(self, res):
        """
        Adds an image to the viewer when loaded, using its filepath as the name.
        """
        img, fpath = res
        # Add the image to the overall dict
        self.image_path_dict[fpath.name] = fpath
        self.viewer.add_image(img, name=fpath.name)
        self.load_img_counter += 1
        num_files = len(self.image_path_dict)
        self.view_img_btn.setText(
            f"Loading...({self.load_img_counter}/{num_files} images loaded)."
        )
        if self.load_img_counter == num_files:
            self.view_img_btn.setText(
                f"All ({self.load_img_counter}/{num_files}) images loaded."
            )

    def _reset_view_btn(self):
        """Reset the view button to be clickable again when done."""
        self.view_img_btn.setEnabled(True)

    def watch_mask_files(self):
        """
        File watcher to watch for new mask files being created during the Nextflow run.

        This is used to update the napari Labels layers with the new masks.

        Currently expects that the slices are stored as .npy files. Deactivates
        when it sees each image has an associated "*_all.npy" file.
        """
        # Wait for at least one image to load as layers if not present
        if not self.viewer.layers:
            time.sleep(0.5)
        # Create the Labels layers for each image
        for fpath in self.image_path_dict.values():
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
                # Filter out files we are not running on
                current_files = [
                    i for i in current_files if Path(i).stem.split("_")[0] in self.image_path_dict
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
                num_images = len(self.image_path_dict)
                # If all images have complete masks, deactivate watcher
                if all(masks_finished) and (len(masks_finished) == num_images):
                    print("Deactivating watcher...")
                    self.watcher_enabled = False

        # Call the nested function
        _watch_mask_files(self)

    def update_masks(self, new_files):
        """
        Update the masks in the napari Labels layers with the new masks found in the last scan.
        """
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            mask_arr = np.load(f)
            # Extract the relevant Labels layer
            split_names = f.stem.split("_masks")
            layer_name = split_names[0] + "_masks"
            label_layer = self.viewer.layers[layer_name]
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

    def create_nxf_box(self):
        """
        Create the widget box containing options for the Nextflow pipeline.
        """
        self.nxf_group = QGroupBox("Nextflow Pipeline:")
        self.nxf_layout = QGridLayout()
        # Create a drop-down box to select the execution profile
        self.nxf_profile_label = QLabel("Execution profile:")
        self.nxf_profile_label.setToolTip(
            "Select the execution profile to use."
        )
        self.nxf_profile_box = QComboBox()
        # Get the available profiles from config dir
        # TODO: This will not work when Nextflow has been separated
        config_dir = Path(__file__).parent / "nextflow" / "profiles"
        avail_confs = [str(i.stem) for i in config_dir.glob("*.conf")]
        self.nxf_profile_box.addItems(avail_confs)
        self.nxf_layout.addWidget(self.nxf_profile_label, 0, 0)
        self.nxf_layout.addWidget(self.nxf_profile_box, 0, 1)
        # Create a button to navigate to a directory to take images from
        self.nxf_btn = QPushButton("Run Pipeline!")
        self.nxf_btn.clicked.connect(self.run_pipeline)
        self.nxf_btn.setToolTip(
            "Run the pipeline with the chosen organelle(s), model, and images."
        )
        self.nxf_layout.addWidget(self.nxf_btn, 1, 0, 1, 2)
        self.nxf_group.setLayout(self.nxf_layout)
        self.layout().addWidget(self.nxf_group)

    def store_img_paths(self):
        """
        Write the image paths of all images to a text file for input to the Nextflow pipeline.
        """
        self.img_list_fpath = Path(__file__).parent / "all_img_paths.txt"
        # Extract the paths for all the stored images
        img_file_paths = self.image_path_dict.values()
        # Write the image paths into a newline-separated text file
        with open(self.img_list_fpath, "w") as output:
            output.write("\n".join([str(i) for i in img_file_paths]))

    def create_nextflow_params(self):
        """
        Create the parameters to pass to the Nextflow pipeline
        """
        if self.model_config is None:
            config_path = self.get_model_config()
        else:
            config_path = self.model_config
        # Extract the current model version selected
        model_type = self.model_version_dropdown.currentText()
        # Construct the params to be given to Nextflow
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
        """
        Construct the model config from the parameter widgets.
        """
        # Get the current dictionary of widgets for selected model
        model_dict_orig = self.model_param_dict[self.selected_model]
        # Get the relevant default params for this model
        default_params = MODEL_INFO[self.selected_model]["params"]
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
            orig_dtype = default_params[param_name].dtype
            model_dict[default_params[param_name].arg] = orig_dtype(
                param_value
            )
        # Extract the model type
        model_type = self.model_version_dropdown.currentText()
        # Define save path for the model config
        config_dir = Path(__file__).parent / "nextflow" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        model_config_fpath = (
            config_dir / f"{self.selected_model}-{model_type}_config.yaml"
        )
        # Save the yaml config
        with open(model_config_fpath, "w") as f:
            yaml.dump(model_dict, f)
        return model_config_fpath

    def run_pipeline(self):
        """
        Run the nextflow pipeline. Checks a task and model are selected, and that images have been loaded. Loads the images to the viewer, and starts the file watcher for the masks.
        """
        # Start with some error checking to ensure that everything has been properly specified
        # Check a task/organelle has been selected
        if self.selected_task is None:
            raise ValueError("No task/organelle has been selected!")
        # Check a model has been selected
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
