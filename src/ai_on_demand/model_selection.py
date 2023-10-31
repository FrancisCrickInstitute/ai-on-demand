from pathlib import Path
from typing import Optional

import napari
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
)
import yaml

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.models import (
    MODEL_INFO,
    MODEL_DISPLAYNAMES,
    TASK_MODELS,
    MODEL_TASK_VERSIONS,
)
from ai_on_demand.utils import format_tooltip, sanitise_name, merge_dicts


class ModelWidget(SubWidget):
    _name = "model"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QVBoxLayout,
    ):
        # Set selection colour
        # Needs to be done before super call
        self.colour_selected = "#F7AD6F"

        super().__init__(
            viewer=viewer,
            title="Model",
            parent=parent,
            layout=layout,
            tooltip="""
Select the model and model variant to use for inference.

Parameters can be modified if setup properly, otherwise a config file can be loaded in whatever format the model takes!
        """,
        )

    def create_box(self, variant: Optional[str] = None):
        # TODO: This will have to become a variant for e.g. fine-tuning
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

        self.layout().addLayout(model_box_layout)

        # Store model config location if given
        self.model_config = None

        # Create container for switching between setting params and loading config
        self.params_config_widget = QWidget()
        self.params_config_layout = QHBoxLayout()
        # Create button for displaying model param options
        self.model_param_btn = QPushButton("Modify Parameters")
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
        self.model_config_btn = QPushButton("Load/Save Config File")
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
        self.layout().addWidget(self.params_config_widget)

        # Create widgets for the two options
        self.create_model_param_widget()
        self.create_model_config_widget()

        self.widget.setLayout(self.layout())

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
        self.parent.selected_model = MODEL_DISPLAYNAMES[model_name]
        # Update the dropdown for the model variants
        self.model_version_dropdown.clear()
        model_versions = MODEL_INFO[self.parent.selected_model]["versions"][
            self.parent.selected_task
        ]
        self.model_version_dropdown.addItems(model_versions)
        self.parent.selected_variant = (
            self.model_version_dropdown.currentText()
        )
        # Update the model params & config widgets for the selected model
        self.update_model_param_config(
            self.parent.selected_model, self.parent.selected_variant
        )

    def on_model_version_select(self):
        # Update tracker for selected model variant/version
        self.parent.selected_variant = (
            self.model_version_dropdown.currentText()
        )
        # Update the model params & config widgets for the selected model variant/version
        self.update_model_param_config(
            self.parent.selected_model, self.parent.selected_variant
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
        self.model_config_load_btn = QPushButton("Select model config")
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
        self.layout().addWidget(self.model_config_widget)

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
        self.layout().addWidget(self.model_param_widget)

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
            param_dict = MODEL_INFO[model_name]["params"][
                self.parent.selected_task
            ]
            # Check if there is a version-specific set of params
            if model_version in param_dict:
                param_dict = param_dict[model_version]
        except KeyError as e:
            raise e("Default model parameters not found!")
        # Construct the unique tuple for this widget
        # NOTE: Likely to create a lot of redundant widgets, but should be light on memory
        # and is the most extendable
        model_task_version = (
            model_name,
            self.parent.selected_task,
            model_version,
        )
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

    def get_model_config(self):
        # First check if there is a config file for this model
        try:
            model_task_dict = MODEL_TASK_VERSIONS[self.parent.selected_model][
                self.parent.selected_task
            ][self.parent.selected_variant]
        except KeyError as e:
            raise Exception(
                f"No config file found for {self.parent.selected_model} ({self.parent.selected_variant}) to segment {self.parent.selected_task}!"
            ) from e
        if "config" in model_task_dict:
            # Set this as the base config
            base_config = model_task_dict["config"]
            # Load this config
            with open(Path(model_task_dict["dir"]) / base_config, "r") as f:
                model_dict = yaml.safe_load(f)
            # If there are parameters to overwrite, insert them into the base
            # NOTE: This requires that the base parameters come from this config!
            # NOTE: This currently does not happen
            default_params = MODEL_INFO[self.parent.selected_model]["params"][
                self.parent.selected_task
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
        # NOTE: We should constantly be updating this, so not needed...
        self.parent.selected_variant = (
            self.model_version_dropdown.currentText()
        )
        # Define save path for the model config
        config_dir = self.parent.subwidgets["nxf"].nxf_base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        model_config_fpath = (
            config_dir
            / f"{self.parent.selected_model}-{sanitise_name(self.parent.selected_variant)}_config.yaml"
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
            (
                self.parent.selected_model,
                self.parent.selected_task,
                self.parent.selected_variant,
            )
        ]
        # Get the relevant default params for this model
        default_params = MODEL_INFO[self.parent.selected_model]["params"][
            self.parent.selected_task
        ]
        # Check if there is a version-specific set of params
        if self.parent.selected_variant in default_params:
            default_params = default_params[self.parent.selected_variant]
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
