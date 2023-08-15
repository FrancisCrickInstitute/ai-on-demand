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
    QScrollArea,
    QProgressBar,
)
from qtpy.QtGui import QPixmap
import qtpy.QtCore
import numpy as np
import skimage.io

import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari.layers import Image

from .models import (
    MODEL_INFO,
    MODEL_DISPLAYNAMES,
    TASK_MODELS,
    MODEL_TASK_VERSIONS,
)
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

        # Set the basepath to watch for masks
        self.mask_base_path = Path(__file__).parent / ".nextflow" / "cache"

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
                self.image_path_dict[Path(img_path).stem] = Path(img_path)
            # Then update the counts of files (and their types) with the extra image
            self.update_file_count()

    def on_layer_removed(self, event):
        """
        Triggered whenever a layer is removed from the viewer.

        Checks if the layer is an image, and if so, removes it from the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = event.value.source.path
            # Remove from the list of images
            if (
                img_path is not None
                and Path(img_path).stem in self.image_path_dict
            ):
                del self.image_path_dict[Path(img_path).stem]
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
        self.model_name_unavail = "No models available!"
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
        self.model_version_dropdown.activated.connect(
            self.on_model_version_select
        )
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
            self.clear_model_param_widget()
            self.set_model_param_widget("init")
            return
        elif model_name == self.model_name_unavail:
            self.clear_model_param_widget()
            self.set_model_param_widget("init")
            self.model_version_dropdown.clear()
            self.model_version_dropdown.addItems([self.model_name_unavail])
            return
        self.selected_model = MODEL_DISPLAYNAMES[model_name]
        # Update the dropdown for the model variants
        self.model_version_dropdown.clear()
        model_versions = MODEL_INFO[self.selected_model]["versions"][
            self.selected_task
        ]
        self.model_version_dropdown.addItems(model_versions)
        self.selected_variant = self.model_version_dropdown.currentText()
        # Update the model params & config widgets for the selected model
        self.update_model_param_config(
            self.selected_model, self.selected_variant
        )

    def on_model_version_select(self):
        # Update tracker for selected model variant/version
        self.selected_variant = self.model_version_dropdown.currentText()
        # Update the model params & config widgets for the selected model variant/version
        self.update_model_param_config(
            self.selected_model, self.selected_variant
        )

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
        self.model_config_layout = QGridLayout()

        # Add the button for loading a config file
        self.model_config_load_btn = QPushButton("Select model config file")
        self.model_config_load_btn.clicked.connect(self.select_model_config)
        self.model_config_load_btn.setToolTip(
            "Select a config file to be used for the selected model."
            "Note that no checking/validation is done on the config file, it is just given to the model."
        )
        self.model_config_layout.addWidget(self.model_config_load_btn, 0, 0)
        # Add a button for clearing the config file
        self.model_config_clear_btn = QPushButton("Clear config selection")
        self.model_config_clear_btn.clicked.connect(self.clear_model_config)
        self.model_config_clear_btn.setToolTip(
            "Clear the selected model config file."
        )
        self.model_config_layout.addWidget(self.model_config_clear_btn, 0, 1)
        # Add a label to display the selected config file (if any)
        self.model_config_label = QLabel("No model config file selected.")
        self.model_config_label.setWordWrap(True)
        self.model_config_layout.addWidget(self.model_config_label, 1, 0, 1, 2)
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
        # Add a widget for when there aren't parameters, and a config is needed
        no_param_widget = QWidget()
        no_param_widget.setLayout(QVBoxLayout())
        no_param_label = QLabel("Cannot modify parameters for this model!\nPlease select a config file.")
        no_param_widget.layout().addWidget(no_param_label)
        self.model_param_widgets_dict["no_param"] = no_param_widget
        # Disable showing widget until selected to view
        self.model_param_widget.setVisible(False)
        self.model_layout.addWidget(self.model_param_widget)

    def update_model_param_config(self, model_name: str, model_version: str):
        """
        Updates the model param and config widgets for a specific model.

        Currently only updates the model widget as the config is now constant.
        """
        # Update the model parameters
        self.update_model_param_widget(model_name, model_version)

    def update_model_param_widget(self, model_name: str, model_version: str):
        """
        Updates the model param widget for a specific model
        """
        # Skip if initial message is still showing and clicked
        if model_name not in MODEL_INFO:
            return
        # Remove the current model param widget
        self.clear_model_param_widget()
        # Extract the default parameters
        try:
            param_dict = MODEL_INFO[model_name]["params"][self.selected_task]
            # Check if there is a version-specific set of params
            if model_version in param_dict:
                param_dict = param_dict[model_version]
        except KeyError as e:
            raise e("Default model parameters not found!")
        # Construct the unique tuple for this widget
        # NOTE: Likely to create a lot of redundant widgets, but should be light on memory
        # and is the most extendable
        model_task_version = (model_name, self.selected_task, model_version)
        # Retrieve the widget for this model if already created
        if model_task_version in self.model_param_widgets_dict:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                model_task_version
            ]
        # If no parameters, use the no_param widget
        elif not param_dict:
            self.curr_model_param_widget = self.model_param_widgets_dict["no_param"]
        # Otherwise construct it
        else:
            self.curr_model_param_widget = self._create_model_params_widget(
                model_task_version, param_dict
            )
            self.model_param_widgets_dict[
                model_task_version
            ] = self.curr_model_param_widget
        # # Set the collapsible box to contain the params for this model
        # self.model_param_layout.addWidget(self.curr_model_param_widget)
        # # Ensure it's visible if the params button is pressed
        # if self.model_param_btn.isChecked():
        #     self.curr_model_param_widget.setVisible(True)
        # else:
        #     self.curr_model_param_widget.setVisible(False)
        # Set the current model param widget
        self.set_model_param_widget()

    def _create_model_params_widget(self, model_task_version, param_dict):
        """
        Creates the widget for a specific model's parameters to swap in and out
        """
        # Create a widget for the model parameters
        model_widget = QWidget()
        model_layout = QGridLayout()
        # Create container for model parameters
        self.model_param_dict[model_task_version] = {}
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
            self.model_param_dict[model_task_version][label] = {
                "label": param_label,
                "value": param_value,
            }
        # Tighten up margins and set layout
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_widget.setLayout(model_layout)
        return model_widget

    def clear_model_param_widget(self):
        # Remove the current model param widget
        self.model_param_layout.removeWidget(self.curr_model_param_widget)
        self.curr_model_param_widget.setParent(None)

    def set_model_param_widget(self, model_task_version=None):
        if model_task_version is not None:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                model_task_version
            ]
        # Set the collapsible box to contain the params for this model
        self.model_param_layout.addWidget(self.curr_model_param_widget)
        # Ensure it's visible if the params button is pressed
        if self.model_param_btn.isChecked():
            self.curr_model_param_widget.setVisible(True)
        else:
            self.curr_model_param_widget.setVisible(False)

    def update_model_box(self, task_name):
        """The model box updates according to what's defined for each task."""
        # Clear and set available models in dropdown
        self.model_dropdown.clear()
        # Check that there is a model available for this task (always should be...)
        if task_name in TASK_MODELS:
            model_names = [
                MODEL_INFO[model]["display_name"]
                for model in TASK_MODELS[task_name]
            ]
        else:
            model_names = [self.model_name_unavail]
        self.model_dropdown.addItems(model_names)
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
        # Reset if dialog cancelled
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

    def clear_model_config(self):
        self.model_config_label.setText("No model config file selected.")
        self.model_config = None

    def create_dir_box(self):
        """
        Create the box for selecting the directory to take images from, and optionally view them.

        Displays the number and types of files found in the selected directory.
        """
        # TODO: Simultaneously allow for drag+dropping, probably a common use pattern
        self.dir_group = QGroupBox("Data Selection:")
        self.dir_layout = QGridLayout()
        self.dir_group.setToolTip(
            "Select a directory to take images from, or select individual images.\n"
            "Images can also be opened, or dragged into napari as normal. The selection will be updated accordingly.\n"
            "Note that all images loaded are additive, unless removed as a layer. The 'Reset selection' button can be used to clear all images.\n"
        )
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
                        self.image_path_dict[img_path.stem] = img_path
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
        self.dir_btn = QPushButton("Select image directory")
        self.dir_btn.clicked.connect(self.browse_imgs_dir)
        self.dir_btn.setToolTip(
            "Select folder/directory of images to use as input to the model.\n"
            "Note that this allows for running the pipeline without having to load images into napari first.\n"
            "Any images loaded into napari will also be used within the pipeline, however."
        )
        self.dir_layout.addWidget(self.dir_btn, 0, 1)
        # Add an output to show the counts
        self.init_file_msg = "No files selected or added to Napari."
        self.img_counts = QLabel(self.init_file_msg)
        self.img_counts.setWordWrap(True)
        self.dir_layout.addWidget(self.img_counts, 1, 0, 1, 2)

        # Add a button for viewing the images within napari
        # Optional as potentially taxing, and not necessary
        self.view_img_btn = QPushButton("View selected images")
        self.view_img_btn.setToolTip(
            "Load selected images into napari to view."
        )
        self.view_img_btn.clicked.connect(self.view_images)
        self.dir_layout.addWidget(self.view_img_btn, 2, 0)
        # Create a button to clear selected directory
        self.clear_dir_btn = QPushButton("Reset selection")
        self.clear_dir_btn.clicked.connect(self.clear_directory)
        self.clear_dir_btn.setToolTip(
            "Reset selection of images (clears all images in the viewer)."
        )
        self.dir_layout.addWidget(self.clear_dir_btn, 2, 1)
        # Add button layout to box layout
        # Sort out layout and add to main widget
        self.dir_group.setLayout(self.dir_layout)
        self.layout().addWidget(self.dir_group)

    def browse_imgs_files(self):
        fnames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more images",
            str(Path.home()),
            "",
        )
        if fnames != []:
            self.update_file_count(paths=fnames)

    def browse_imgs_dir(self):
        """
        Opens a dialog for selecting a directory that contains images to segment.
        """
        # TODO: Load multiple directories - https://stackoverflow.com/a/28548773/9963224
        # Quite the pain, and potentially brittle if Qt backend changes
        result = QFileDialog.getExistingDirectory(
            self, caption="Select image directory", directory=None
        )
        if result != "":
            self.update_file_count(paths=list(Path(result).glob("*")))

    def clear_directory(self):
        """
        Clears the selected directory and resets the image counts.
        """
        # Reset selected images and their paths
        self.image_path_dict = {}
        # Reset image count text
        self.img_counts.setText(self.init_file_msg)
        # Reset the images loaded button text
        self.view_img_btn.setText("View selected images")
        # Remove Image layers from napari viewer
        img_layers = [i for i in self.viewer.layers if isinstance(i, Image)]
        for layer in img_layers:
            self.viewer.layers.remove(layer)

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
        txt = "Selected "
        # Add paths to the overall list if specific ones need adding
        if paths is not None:
            for img_path in paths:
                img_path = Path(img_path)
                self.image_path_dict[img_path.stem] = img_path
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
        txt += f" file{'s' if sum(extension_counts.values()) > 1 else ''}."
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
            Path(i.name).stem
            for i in self.viewer.layers
            if isinstance(i, Image)
        ]
        imgs_to_load = [
            v for k, v in self.image_path_dict.items() if k not in viewer_imgs
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
        self.image_path_dict[fpath.stem] = fpath
        self.viewer.add_image(img, name=fpath.stem)
        self.load_img_counter += 1
        self.view_img_btn.setText(
            f"Loading...({self.load_img_counter} image{'s' if self.load_img_counter > 1 else ''} loaded)."
        )
        img_layers = [i for i in self.viewer.layers if isinstance(i, Image)]
        # Only change text when we have as many image layers as images
        if len(img_layers) == len(self.image_path_dict):
            self.view_img_btn.setText("All images loaded.")
        # Update the progress bar range (just in case the image wasn't loaded in time)
        if img.ndim > 2:
            self.progress_bar_dict[fpath.stem].setRange(0, img.shape[-3])
            self.progress_bar_dict[fpath.stem].setValue(0)

    def _reset_view_btn(self):
        """Reset the view button to be clickable again when done."""
        self.view_img_btn.setEnabled(True)
        # Also reset the viewer itself
        self.viewer.reset_view()

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
            name = (
                f"{prefix}_masks_{self.selected_model}-{self.selected_variant}"
            )
            # Add a Labels layer for this file
            self.viewer.add_labels(
                np.zeros(img_shape, dtype=int), name=name, visible=False
            )
            # Move this layer to the top
            self.viewer.layers.move(self.viewer.layers.index(name), -1)
        # Construct the proper mask path
        self.mask_path = (
            self.mask_base_path
            / f"{self.selected_model}"
            / f"{self.selected_variant}_masks"
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
                    i
                    for i in current_files
                    if Path(i).stem.split("_masks_")[0] in self.image_path_dict
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
            mask_layer_name = f"{f.stem.split('_masks_')[0]}_masks_{self.selected_model}-{self.selected_variant}"
            label_layer = self.viewer.layers[mask_layer_name]
            # Insert mask data
            label_layer.data = mask_arr
            label_layer.visible = True
            slice_num = f.stem.split("_")[-1]
            # Switch viewer to latest slice
            if slice_num == "all":
                slice_num = label_layer.data.shape[0] - 1
            else:
                slice_num = int(slice_num)
            self.viewer.dims.set_point(0, slice_num)
            # Increment the associated progress bar
            print(f.stem, slice_num + 1)
            self.progress_bar_dict[f"{f.stem.split('_masks_')[0]}"].setValue(
                slice_num + 1
            )

    def on_click_export(self):
        """
        Callback for when the export button is clicked. Opens a dialog to select a directory to save the masks to.
        """
        export_dir = QFileDialog.getExistingDirectory(
            self, caption="Select directory to save masks", directory=None
        )
        # Get all the mask layers
        mask_layers = []
        for img_name in self.image_path_dict:
            layer_name = f"{img_name}_masks_{self.selected_model}-{self.selected_variant}"
            if layer_name in self.viewer.layers:
                mask_layers.append(self.viewer.layers[layer_name])
        # Extract the data from each of the layers, and save the result in the given folder
        for mask_layer in mask_layers:
            np.save(Path(export_dir) / f"{layer_name}.npy", mask_layer.data)

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

        # Add a button for exporting masks
        self.export_masks_btn = QPushButton("Export masks")
        self.export_masks_btn.clicked.connect(self.on_click_export)
        self.export_masks_btn.setToolTip(
            "Export the segmentation masks to a directory."
        )
        self.export_masks_btn.setEnabled(False)
        self.nxf_layout.addWidget(self.export_masks_btn, 2, 0, 1, 1)

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
        self.selected_variant = self.model_version_dropdown.currentText()
        # Construct the params to be given to Nextflow
        nxf_params = {}
        nxf_params["img_dir"] = str(self.img_list_fpath)
        nxf_params["model"] = self.selected_model
        nxf_params["model_config"] = config_path
        nxf_params["model_type"] = self.selected_variant
        nxf_params["task"] = self.selected_task
        # Extract the model checkpoint location and location type
        checkpoint_info = MODEL_TASK_VERSIONS[self.selected_model][
            self.selected_task
        ][self.selected_variant]
        if "url" in checkpoint_info:
            nxf_params["model_chkpt_type"] = "url"
            nxf_params["model_chkpt_loc"] = checkpoint_info["url"]
            nxf_params["model_chkpt_fname"] = checkpoint_info["filename"]
        elif "dir" in checkpoint_info:
            nxf_params["model_chkpt_type"] = "dir"
            nxf_params["model_chkpt_loc"] = checkpoint_info["dir"]
            nxf_params["model_chkpt_fname"] = checkpoint_info["filename"]
        # TODO: Implement profiles for this to configure SLURM
        nxf_params["executor"] = self.nxf_profile_box.currentText()
        return nxf_params

    def get_model_config(self):
        """
        Construct the model config from the parameter widgets.
        """
        # Get the current dictionary of widgets for selected model
        model_dict_orig = self.model_param_dict[
            (self.selected_model, self.selected_task, self.selected_variant)
        ]
        # Get the relevant default params for this model
        default_params = MODEL_INFO[self.selected_model]["params"][
            self.selected_task
        ]
        # Check if there is a version-specific set of params
        if self.selected_variant in default_params:
            default_params = default_params[self.selected_variant]
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
        self.selected_variant = self.model_version_dropdown.currentText()
        # Define save path for the model config
        config_dir = Path(__file__).parent / "nextflow" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        model_config_fpath = (
            config_dir
            / f"{self.selected_model}-{self.selected_variant}_config.yaml"
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
        # Ensure the export masks button is disabled
        self.export_masks_btn.setEnabled(False)
        # If a previous progress bar widget exists, remove it
        if hasattr(self, "progress_bar_widget"):
            print("Removing progress bars")
            self.layout().removeWidget(self.progress_bar_widget)
            self.progress_bar_widget.setParent(None)
        # Create the progress bars
        self.create_progress_bars()
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
            # TODO: Will not work when nextflow is separated, will need to switch to call from repo
            nextflow_script = Path(__file__).parent / "nextflow" / "main.nf"
            exec_str = f"nextflow run {str(nextflow_script)}"
            # Add the command line arguments
            for k, v in nxf_params.items():
                exec_str += f" --{k}={v}"
            # Add the execution profile
            exec_str += f" -profile {self.nxf_profile_box.currentText()}"
            exec_str += " -process.echo true"
            # exec_str += " -with-dag flowchart.png"
            # TODO: Add '-bg' to run in background?
            # TODO: Add '-resume' to resume from last run just in case?
            # Run the pipeline!
            subprocess.run(
                exec_str, shell=True, cwd=Path(__file__).parent, check=True
            )

        _run_pipeline(self, nxf_params)

    def _pipeline_finish(self):
        # Add a notification that the pipeline has finished
        show_info("Pipeline finished!")
        # Reset the run pipeline button
        self.nxf_btn.setText("Run Pipeline!")
        self.nxf_btn.setEnabled(True)
        # Enable the Export Masks button
        self.export_masks_btn.setEnabled(True)

    def _pipeline_fail(self, exc):
        show_info("Pipeline failed! See terminal for details")
        print(exc)
        # Reset the run pipeline button
        self.nxf_btn.setText("Run Pipeline!")
        self.nxf_btn.setEnabled(True)
        raise exc

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
