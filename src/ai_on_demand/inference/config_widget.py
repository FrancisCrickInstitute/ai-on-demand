from typing import Optional

from napari._qt.qt_resources import QColoredSVGIcon
from aiod_registry import TASK_NAMES
import napari
import yaml
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog
)
from ai_on_demand.utils import format_tooltip, get_plugin_cache
from ai_on_demand.widget_classes import SubWidget


class ConfigWidget(SubWidget):
    _name = "config"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Configuration",
            parent=parent,
            layout=layout,
            tooltip="Load and save configurations",
            **kwargs,
        )

    def create_box(self, variant: Optional[str] = None):
        """
        Create the box for loading and saving configurations.
        """
        # Create box for the custom config settings
        self.config_box = QGroupBox("Config Settings")
        self.config_box.setToolTip(
            format_tooltip("save custom configs settings which will automatically fill in all options in the ai-od pugin")
        )
        self.config_layout = QGridLayout()
        self.config_box.setLayout(self.config_layout)

        self.config_name_label = QLabel("Config name:")
        self.config_name_input = QLineEdit(placeholderText="config_A")

        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.on_load_config)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.setDisabled(True)
        self.save_config_button.setToolTip(
            format_tooltip("Saving becomes available after running pipeline once")
        )
        self.save_config_button.clicked.connect(self.on_save_config)


        # TODO: probably not needed but can have further info about how save config works
        # # Add an icon for information about the selected model
        # self.model_info_icon = QPushButton("")
        # self.model_info_icon.setIcon(
        #     QColoredSVGIcon.from_resources("help").colored(theme="dark")
        # )
        # # Fix the size, and set the icon as a percentage of this
        # self.model_info_icon.setFixedSize(30, 30)
        # self.model_info_icon.setIconSize(self.model_info_icon.size() * 0.65)
        # self.model_info_icon.setToolTip(
        #     format_tooltip("Information about saving configs")
        # )
        # self.config_layout.addWidget(self.model_info_icon, 0, 3)

        self.config_layout.addWidget(self.config_name_label, 0, 0)
        self.config_layout.addWidget(self.config_name_input, 0, 1)
        self.config_layout.addWidget(self.load_config_button, 1, 0)
        self.config_layout.addWidget(self.save_config_button, 0, 2)

        self.inner_layout.addWidget(self.config_box)
        

    def on_click_task(self):
        """
        Callback for when a task button is clicked.

        Updates the model box to show only the models available for the selected task.
        """
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
                self.parent.selected_task = task_name
        # Update the model box for the selected task
        self.parent.subwidgets["model"].update_model_box(
            self.parent.selected_task
        )
    
    def on_load_config(self):
        config_dir, _ = get_plugin_cache()
        config_path_and_filter = QFileDialog.getOpenFileName(self, "select a config file", str(config_dir), "YAML Files (*.yaml *.yml)")
        config_path = config_path_and_filter[0]
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            if config_data:
                self.parent.load_config_file(config_data)
    
    def enable_save_config(self): 
        self.save_config_button.setDisabled(False)

    def on_save_config(self):
        config_name = self.config_name_input.text().strip()
        if not config_name:
            config_name = "aiod-inference-config"

        self.parent.store_config(config_name)
