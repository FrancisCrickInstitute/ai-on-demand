from pathlib import Path
from typing import Optional

import napari
from napari._qt.qt_resources import get_stylesheet
from qtpy.QtCore import Qt
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
    QDialog,
    QTextEdit,
)
import yaml

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import (
    format_tooltip,
    sanitise_name,
    merge_dicts,
    calc_param_hash,
    load_config,
)


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
        # Extract the model info from all manifests
        self.extract_model_info()

        # Track whether the model defaults have been changed
        self.changed_defaults = False

    def extract_model_info(self):
        # Initialise model-related attributes
        # Easy access to the display name for each model
        self.base_to_display = {}
        self.display_to_base = {}
        # Dict of available models, and model versions, for each task
        self.versions_per_task = {}
        # Dict of model params for each model version, specific to each task
        self.model_version_tasks = {}

        # Extract the model info from all manifests
        for model_manifest in self.parent.all_manifests:
            # Get the short and display names
            base_name = model_manifest.short_name
            self.base_to_display[base_name] = model_manifest.name
            self.display_to_base[model_manifest.name] = base_name
            # Get each version
            for version_name, version in model_manifest.versions.items():
                # Get the tasks for this version
                for task_name, task in version.tasks.items():
                    # Add this task if not yet seen
                    if task_name not in self.versions_per_task:
                        self.versions_per_task[task_name] = {}
                    # Add this base model if not yet seen for this task
                    if base_name not in self.versions_per_task[task_name]:
                        self.versions_per_task[task_name][base_name] = []
                    # Add this version, for this base model, under this task
                    self.versions_per_task[task_name][base_name].append(
                        version_name
                    )
                    # Store this model-version-task for easy access to params and config
                    self.model_version_tasks[
                        (task_name, base_name, version_name)
                    ] = task

    def create_box(self, variant: Optional[str] = None):
        # TODO: This will have to become a variant for e.g. fine-tuning
        model_box_layout = QGridLayout()
        # Create a label for the dropdown
        model_label = QLabel("Select model:")
        # Dropdown of available models
        self.model_dropdown = QComboBox()
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
        self.model_version_dropdown.addItems(["Select a model first!"])
        self.model_version_dropdown.activated.connect(
            self.on_model_version_select
        )
        model_box_layout.addWidget(model_version_label, 1, 0)
        model_box_layout.addWidget(self.model_version_dropdown, 1, 1, 1, 2)

        # Add an icon for information about the selected model
        self.model_info_icon = QPushButton("🔍")
        # Fix the size, and set the icon as a percentage of this
        self.model_info_icon.setFixedSize(30, 30)
        self.model_info_icon.setIconSize(self.model_info_icon.size() * 0.65)
        self.model_info_icon.setToolTip(
            format_tooltip("Information about the selected model.")
        )
        self.model_info_icon.clicked.connect(self.on_model_info)
        model_box_layout.addWidget(self.model_info_icon, 0, 3, 2, 1)

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
        # Get the shorthand name from the model display name
        self.parent.selected_model = self.display_to_base[model_name]
        # Update the dropdown for the model variants
        self.model_version_dropdown.clear()
        # Extract the model versions for this model for this task
        model_versions = self.versions_per_task[self.parent.selected_task][
            self.parent.selected_model
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
        # TODO: Figure out when this occurs, and why it isn't model_name == model_name_init, or model_name == model_name_unavail
        if model_name not in self.base_to_display:
            return
        # Remove the current model param widget
        self.clear_model_param_widget()
        # Construct the unique tuple for this widget
        task_model_version = (
            self.parent.selected_task,
            model_name,
            model_version,
        )
        # Extract the default parameters for this model-version-task
        param_list = self.model_version_tasks[task_model_version].params
        # NOTE: Likely to create a lot of redundant widgets, but should be light on memory
        # and is the most extendable
        # Retrieve the widget for this model if already created
        if task_model_version in self.model_param_widgets_dict:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                task_model_version
            ]
        # If no parameters, use the no_param widget
        elif param_list is None:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                "no_param"
            ]
        # Otherwise, construct it
        else:
            self.curr_model_param_widget = self._create_model_params_widget(
                task_model_version, param_list
            )
            self.model_param_widgets_dict[task_model_version] = (
                self.curr_model_param_widget
            )
        # Set the current model param widget
        self.set_model_param_widget()

    def _create_model_params_widget(
        self, task_model_version: tuple[str, str, str], param_list: list
    ):
        """
        Creates the widget for a specific model's parameters to swap in and out
        """
        # Create a widget for the model parameters
        model_widget = QWidget()
        model_layout = QGridLayout()
        # Create container for model parameters
        self.model_param_dict[task_model_version] = {}
        # Add the default model parameters
        for i, model_param in enumerate(param_list):
            # Create labels for each of the model parameters
            param_label = QLabel(f"{model_param.name}:")
            param_label.setToolTip(format_tooltip(model_param.tooltip))
            model_layout.addWidget(param_label, i, 0)
            # Add the model parameter(s)
            param_values = model_param.value
            # Widget added depends on the input
            # True/False -> Checkbox
            if param_values is True or param_values is False:
                param_val_widget = QCheckBox()
                # Checked if default param value is True, unchecked if False
                if param_values:
                    param_val_widget.setChecked(True)
                else:
                    param_val_widget.setChecked(False)
                param_val_widget.stateChanged.connect(self.on_param_changed)
            # List -> ComboBox
            elif isinstance(param_values, list):
                param_val_widget = QComboBox()
                param_val_widget.addItems([str(i) for i in param_values])
                param_val_widget.currentIndexChanged.connect(
                    self.on_param_changed
                )
            # Int/float/str -> LineEdit
            elif isinstance(param_values, (int, float, str)):
                param_val_widget = QLineEdit()
                param_val_widget.setText(str(param_values))
                param_val_widget.textChanged.connect(self.on_param_changed)
            else:
                # Should be handled on the Pydantic side
                raise ValueError(
                    f"Model parameter {model_param.name} has invalid type {type(param_values)}"
                )
            model_layout.addWidget(param_val_widget, i, 1)
            # Store for later retrieval when saving the config
            self.model_param_dict[task_model_version][model_param.name] = {
                "label": param_label,
                "value": param_val_widget,
            }
        # Tighten up margins and set layout
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_widget.setLayout(model_layout)
        return model_widget

    def on_param_changed(self):
        self.changed_defaults = True

    def clear_model_param_widget(self):
        # Remove the current model param widget
        self.model_param_layout.removeWidget(self.curr_model_param_widget)
        self.curr_model_param_widget.setParent(None)

    def set_model_param_widget(self, task_model_version=None):
        if task_model_version is not None:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                task_model_version
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
        # Check that there is a model available for this task
        if task_name in self.versions_per_task:
            model_names = sorted(
                [
                    self.base_to_display[i]
                    for i in self.versions_per_task[task_name].keys()
                ]
            )
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
            "Configs (*.yaml *.yml *.json)",
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
        """
        Need to construct the final model configuration.

        If a model is found in the schema, this is loaded in as the base config.

        If a config file is loaded, this simply replaces the base config. We do not merge them!

        If a config file is not loaded, if GUI parameters exist we merge them into the base config.
        """
        # First check if there is a config file for this model
        # Use the executed model, variant, and task to get the version
        # NOTE: Almost 100% sure same as selected at this point
        task_model_version = self.get_task_model_variant(executed=True)
        model_version = self.model_version_tasks[task_model_version]

        if self.model_config is not None:
            # Load the config file
            model_dict = load_config(self.model_config)
        elif model_version.config_path is not None:
            # Set this as the base config
            model_dict = load_config(Path(model_version.config_path))
            # Merge in the GUI params if they exist and have changed
            if model_version.params is not None and self.changed_defaults:
                gui_dict = self.create_config_params(
                    task_model_version=task_model_version
                )
                model_dict = merge_dicts(model_dict, gui_dict)
        # No loaded config, no schema config
        else:
            # NOTE: May fail if there are no GUI params, on top of no schema or loaded config
            model_dict = self.create_config_params(
                task_model_version=task_model_version
            )
        # Get the unique hash to this set of parameters
        self.model_param_hash = calc_param_hash(model_dict)
        # Save the model config
        model_config_fpath = self.save_model_config(model_dict)
        return model_config_fpath

    def save_model_config(self, model_dict: dict):
        # Define save path for the model config
        config_dir = self.parent.subwidgets["nxf"].nxf_base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        task_model_variant_name = self.get_task_model_variant_name()

        # Name the config using the same task-model-version convention as the masks
        model_config_fpath = (
            config_dir
            / f"{task_model_variant_name}_config_{self.model_param_hash}.yaml"
        )
        # Save the yaml config
        with open(model_config_fpath, "w") as f:
            yaml.dump(model_dict, f)
        return model_config_fpath

    def create_config_params(self, task_model_version: Optional[tuple] = None):
        """
        Construct the model config from the parameter widgets.

        If no task_model_version is given, the executed model, variant, and task are used.
        """
        # Get the current dictionary of widgets for selected model
        if task_model_version is None:
            task_model_version = self.get_task_model_variant(executed=True)
        model_dict_orig = self.model_param_dict[task_model_version]
        # Get the relevant default params for this model
        default_params = self.model_version_tasks[task_model_version].params
        # Reformat the dict to pipe into downstream model run scripts
        model_dict = {}
        # Extract params from model param widgets
        for orig_param, (param_name, sub_dict) in zip(
            default_params, model_dict_orig.items()
        ):
            # Dicts maintain insertion order, so this should be fine, but double-check
            assert orig_param.name == param_name
            if isinstance(sub_dict["value"], QLineEdit):
                param_value = sub_dict["value"].text()
            elif isinstance(sub_dict["value"], QComboBox):
                param_value = sub_dict["value"].currentText()
            elif isinstance(sub_dict["value"], QCheckBox):
                param_value = sub_dict["value"].isChecked()
            else:
                raise NotImplementedError
            # Extract the original/intended dtype and cast what's in the box
            model_dict[orig_param.arg_name] = orig_param._dtype(param_value)
        return model_dict

    def get_task_model_variant(self, executed: bool = True):
        if executed:
            task, model, version = (
                self.parent.executed_task,
                self.parent.executed_model,
                self.parent.executed_variant,
            )
        else:
            task, model, version = (
                self.parent.selected_task,
                self.parent.selected_model,
                self.parent.selected_variant,
            )
        return (task, model, version)

    def get_task_model_variant_name(self, executed: bool = True):
        task, model, version = self.get_task_model_variant(executed)
        return f"{task}-{model}-{sanitise_name(version)}"

    def on_model_info(self):
        """
        Callback for when the model info button is clicked.

        Opens a dialog box with information about the selected model.
        """
        # Extract the model info from the manifest
        task_model_version = self.get_task_model_variant(executed=False)
        model_version = self.model_version_tasks[task_model_version]
        # Extract info from the schema if present
        # Need to get the parent model this version came from
        full_manifest = self.parent.all_manifests[task_model_version[1]]
        # Construct the message
        model_info = f"""Model: {self.base_to_display[task_model_version[1]]}
Version: {task_model_version[2]}
{full_manifest.metadata}"""
        # Add the parameters if they exist
        # TODO: Reformat to include tooltips
        if model_version.params is not None:
            model_info += "\nParameters:"
            for param in model_version.params:
                model_info += f"\n- {param.name} (default={param.value})"
                if param.tooltip is not None and param.tooltip != "":
                    model_info += f"\n\t{param.tooltip}"
        # Add the config path if it exists
        if model_version.config_path is not None:
            model_info += f"\nConfig path: {model_version.config_path}"

        self.model_window = ModelInfoWindow(self, model_info=model_info)
        self.model_window.show()


class ModelInfoWindow(QDialog):
    def __init__(self, parent=None, model_info: str = ""):
        super().__init__(parent)

        # Set style/look to be same as Napari
        self.setStyleSheet(get_stylesheet("dark"))
        # Set the layout
        self.layout = QVBoxLayout()
        # Set the window title
        self.setWindowTitle("Model Information")
        # Add the info label
        self.info_label = QTextEdit()
        # No text wrapping (horizontal scroll)
        self.info_label.setLineWrapMode(QTextEdit.NoWrap)
        # Make the text selectable, but not editable
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setText(model_info)
        self.info_label.setMinimumSize(500, 500)

        self.layout.addWidget(self.info_label)
        self.setLayout(self.layout)
