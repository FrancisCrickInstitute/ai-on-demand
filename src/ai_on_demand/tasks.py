from typing import Optional

import napari
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QRadioButton,
)

from ai_on_demand.widget_classes import SubWidget

TASK_NAMES = {
    "mito": "Mitochondria",
    "er": "Endoplasmic Reticulum",
    "ne": "Nuclear Envelope",
    "everything": "Everything!",
}


class TaskWidget(SubWidget):
    _name = "task"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        super().__init__(
            viewer=viewer,
            title="Select Organelle to Segment",
            parent=parent,
            layout=layout,
        )

    def create_box(self, variant: Optional[str] = None):
        """
        Create the box for selecting the task (i.e. organelle) to segment.
        """
        # Define and set the buttons for the different tasks
        # With callbacks to change other options accoridngly
        self.task_buttons = {}
        for name, label in TASK_NAMES.items():
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            btn.clicked.connect(self.on_click_task)
            self.layout().addWidget(btn)
            self.task_buttons[name] = btn

        self.widget.setLayout(self.layout())

    def on_click_task(self):
        """
        Callback for when a task button is clicked.

        Updates the model box to show only the models available for the selected task.
        """
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
                # TODO: Temporary patch, maybe not best approach as this should be self-contained and interfaced
                self.parent.selected_task = task_name
        # Update the model box for the selected task
        self.parent.update_model_box(self.parent.selected_task)
