from pathlib import Path
from typing import Union
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel, QLineEdit, QRadioButton, QGroupBox
from qtpy.QtGui import QPixmap
import qtpy.QtCore

import napari

# Available models, with displayable names
MODELS = {
    "unet": "UNET",
    "sam": "Segment Anything"
}
# Specify which models are available for each task
MODEL_DICT = {
    "mito": [k for k in MODELS if k != "sam"],
    "er": [k for k in MODELS if k != "sam"],
    "ne": [k for k in MODELS if k != "sam"],
    "sam": ["sam"],
}

class AIOnDemand(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Set overall layout for widget
        self.setLayout(QVBoxLayout())

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        # Create radio buttons for selecting task (i.e. organelle)
        self.create_organelle_box()

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.create_model_box()

    def create_organelle_box(self):
        # Define the box and layout
        self.task_group = QGroupBox("Select organelle to segment:")
        self.task_layout = QVBoxLayout()
        # Define and set the buttons for the different tasks
        # With callbacks to change other options accoridngly
        self.task_buttons = {}
        for name, label in zip(
            ["mito", "er", "ne", "sam"],
            ["Mitochondria", "Endoplasmic Reticulum", "Nuclear Envelope", "Everything!"]
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
        self.model_sam = QRadioButton("Segment Anything")
        self.model_buttons = {}
        for name, label in MODELS.items():
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            # Necessary to allow for unchecking
            btn.setAutoExclusive(False)
            self.model_layout.addWidget(btn)
            self.model_buttons[name] = btn
        self.model_group.setLayout(self.model_layout)
        self.layout().addWidget(self.model_group)

    def update_model_box(self):
        '''The model box updates according to what's defined for each task
        '''
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
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

    def abspath(self, root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    def _on_click(self):
        import nextflow

        nxf_path = self.abspath(__file__, 'nextflow/main.nf')
        nxf_config_path = self.abspath(__file__, 'nextflow/nextflow.config')

        pipeline1 = nextflow.Pipeline(nxf_path, config=nxf_config_path)

        print(nxf_path)
        print(nxf_config_path)
        print(pipeline1)

        execution = pipeline1.run()

        print(execution.status)

        print(execution.stdout)
