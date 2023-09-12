from pathlib import Path
import time
import yaml

import napari
from napari.qt.threading import thread_worker
import numpy as np
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QGroupBox,
    QComboBox,
    QCheckBox,
)

from ai_on_demand.models import (
    MODEL_INFO,
    MODEL_DISPLAYNAMES,
    TASK_MODELS,
    MODEL_TASK_VERSIONS,
)
from ai_on_demand.tasks import TaskWidget
from ai_on_demand.data_selection import DataWidget
from ai_on_demand.nxf import NxfWidget
from ai_on_demand.utils import sanitise_name, merge_dicts, format_tooltip
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
        self.selected_task = None
        self.selected_model = None

        # Set the basepath to watch for masks
        self.mask_base_path = Path(__file__).parent / ".nextflow" / "cache"

        # Set selection colour
        self.colour_selected = "#F7AD6F"

        # Create radio buttons for selecting task (i.e. organelle)
        self.register_widget(TaskWidget(viewer=self.viewer, parent=self))

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.create_model_box()

        # Create the box for selecting the directory, showing img count etc.
        self.register_widget(DataWidget(viewer=self.viewer, parent=self))

        # Add the button for running the Nextflow pipeline
        self.register_widget(
            NxfWidget(viewer=self.viewer, parent=self, pipeline="inference")
        )

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
            format_tooltip(
                """
        Select the model version to use. Versions can vary either by intended functionality, or are specifically for reproducibility.
        """
            )
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
            format_tooltip(
                "Open options for modifying model parameters directly in napari."
            )
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
            format_tooltip(
                "Open options for loading a config file to pass to the model."
            )
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
            format_tooltip(
                """
            Select a config file to be used for the selected model.
            Note that no checking/validation is done on the config file, it is just given to the model.
        """
            )
        )
        self.model_config_layout.addWidget(self.model_config_load_btn, 0, 0)
        # Add a button for clearing the config file
        self.model_config_clear_btn = QPushButton("Clear config selection")
        self.model_config_clear_btn.clicked.connect(self.clear_model_config)
        self.model_config_clear_btn.setToolTip(
            format_tooltip("Clear the selected model config file.")
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
        no_param_label = QLabel(
            "Cannot modify parameters for this model!\nPlease select a config file."
        )
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
            self.curr_model_param_widget = self.model_param_widgets_dict[
                "no_param"
            ]
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
            param_label.setToolTip(format_tooltip(model_param.tooltip))
            model_layout.addWidget(param_label, i, 0)
            # Add the model parameter(s)
            param_values = model_param.values
            # Widget added depends on the input
            # True/False -> Checkbox
            if param_values is True or param_values is False:
                param_val_widget = QCheckBox()
                if param_values:
                    param_val_widget.setChecked(True)
                else:
                    param_val_widget.setChecked(False)
            # List -> ComboBox
            elif isinstance(param_values, list):
                param_val_widget = QComboBox()
                param_val_widget.addItems([str(i) for i in param_values])
            # Int/float -> LineEdit
            elif isinstance(param_values, (int, float, str)):
                param_val_widget = QLineEdit()
                param_val_widget.setText(str(param_values))
            else:
                raise ValueError(
                    f"Model parameter {label} has invalid type {type(param_values)}"
                )
            model_layout.addWidget(param_val_widget, i, 1)
            # Store for later retrieval when saving the config
            self.model_param_dict[model_task_version][label] = {
                "label": param_label,
                "value": param_val_widget,
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
        for fpath in self.subwidgets["data"].image_path_dict.values():
            # If images still not loaded, add dummy array
            try:
                img_shape = self.viewer.layers[f"{fpath.name}"].data.shape
            except KeyError:
                img_shape = (1000, 1000)
            # Set the layer name for the mask following convention (i.e. the Segment-Flow pipeline output)
            name = self._get_mask_name(fpath.stem)
            # Add a Labels layer for this file
            self.viewer.add_labels(
                np.zeros(img_shape, dtype=int), name=name, visible=False
            )
            # Move this layer to the top
            self.viewer.layers.move(self.viewer.layers.index(name), 0)
        # Construct the proper mask path
        self.mask_dir_path = (
            self.mask_base_path
            / f"{self.selected_model}"
            / f"{sanitise_name(self.selected_variant)}_masks"
        )

        # NOTE: Wrapper as self/class not available at runtime
        @thread_worker(connect={"yielded": self.update_masks})
        def _watch_mask_files(self):
            # Enable the watcher
            print("Activating watcher...")
            self.watcher_enabled = True
            # Initialize empty container for storing mask filepaths
            self.mask_fpaths = []
            # Loop and yield any changes infinitely while enabled
            while self.watcher_enabled:
                # Get all files
                current_files = list(self.mask_dir_path.glob("*.npy"))
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

    def _get_mask_name(self, stem: str):
        return f"{stem}_masks_{self.selected_model}-{sanitise_name(self.selected_variant)}"

    def update_masks(self, new_files):
        """
        Update the masks in the napari Labels layers with the new masks found in the last scan.
        """
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            mask_arr = np.load(f)
            # Check if the mask layer has been renamed
            prefix = f.stem.split("_masks_")[0]
            # Extract the relevant Labels layer
            mask_layer_name = self._get_mask_layer_name(prefix)
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
            # Increment the associated progress bar
            # self.progress_bar_dict[f"{f.stem.split('_masks_')[0]}"].setValue(
            #     slice_num + 1
            # )

    def get_model_config(self):
        # First check if there is a config file for this model
        model_task_dict = MODEL_TASK_VERSIONS[self.selected_model][
            self.selected_task
        ][self.selected_variant]
        if "config" in model_task_dict:
            # Set this as the base config
            base_config = model_task_dict["config"]
            # Load this config
            with open(Path(model_task_dict["dir"]) / base_config, "r") as f:
                model_dict = yaml.safe_load(f)
            # If there are parameters to overwrite, insert them into the base
            # NOTE: This requires that the base parameters come from this config!
            # NOTE: This currently does not happen
            default_params = MODEL_INFO[self.selected_model]["params"][
                self.selected_task
            ]
            if default_params:
                model_params = self.create_config_params()
                # TODO: Need to test this
                model_dict = merge_dicts(model_dict, model_params)
        # Otherwise, just extract from the parameters
        else:
            model_dict = self.create_config_params()
        # Save the model config
        model_config_fpath = self.save_model_config(model_dict)
        return model_config_fpath

    def save_model_config(self, model_dict):
        # Extract the model type
        self.selected_variant = self.model_version_dropdown.currentText()
        # Define save path for the model config
        config_dir = Path(__file__).parent / "nextflow" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        model_config_fpath = (
            config_dir
            / f"{self.selected_model}-{sanitise_name(self.selected_variant)}_config.yaml"
        )
        # Save the yaml config
        with open(model_config_fpath, "w") as f:
            yaml.dump(model_dict, f)
        return model_config_fpath

    def create_config_params(self):
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
                param_value = sub_dict["value"].currentText()
            elif isinstance(sub_dict["value"], QCheckBox):
                param_value = sub_dict["value"].isChecked()
            else:
                raise NotImplementedError
            # Extract the original/intended dtype and cast what's in the box
            orig_dtype = default_params[param_name].dtype
            model_dict[default_params[param_name].arg] = orig_dtype(
                param_value
            )
        return model_dict
