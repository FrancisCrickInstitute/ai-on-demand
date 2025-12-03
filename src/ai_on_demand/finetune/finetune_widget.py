import napari
from .run_finetuning import finetune
from typing import Optional
from napari.qt.threading import thread_worker

from aiod_registry import add_model_local
from qtpy.QtWidgets import (
    QWidget,
    QGridLayout,
    QLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
)
from ai_on_demand.widget_classes import MainWidget, SubWidget, QGroupBox


class Finetune(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer,
            title="Finetuning",
            tooltip="""
            Finetune existing models
                         """,
        )
        self.register_widget(FinetuneWidget(viewer=self.viewer, parent=self))


class FinetuneWidget(SubWidget):
    _name = "finetune"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        super().__init__(
            viewer=viewer,
            title="Finetune",
            parent=parent,
            layout=layout,
            tooltip=parent.tooltip,
        )

    def create_box(self, variant: Optional[str] = None):
        self.finetune_box = QGroupBox("Finetune Model")

        self.finetune_layout = QGridLayout()
        self.finetune_box.setLayout(self.finetune_layout)

        self.train_dir = QLineEdit(placeholderText="Train directory")
        self.model_dir = QLineEdit(placeholderText="Model directory")
        self.finetune_layers = QComboBox()
        self.finetune_layers.addItems(
            [
                "none",
                "layer1",
                "layer2",
                "layer3",
                "layer4",
                "all",
            ]  # what would happen if we did none?
        )
        self.epochs = QSpinBox()
        self.epochs.setRange(
            0, 1000
        )  # would we every do finetuning for more than 1000 epochs?!
        self.epochs.setValue(5)
        self.save_dir = QLineEdit(
            placeholderText="e.g. User/Desktop/finetuned_model"
        )
        self.model_save_name = QLineEdit(
            placeholderText="Name you finetuned model"
        )

        self.finetune_btn = QPushButton("Run Finetuning")
        self.finetune_btn.clicked.connect(self.run_finetuning)

        self.finetune_layout.addWidget(QLabel("Train directory:"), 0, 0)
        self.finetune_layout.addWidget(self.train_dir, 0, 1)

        self.finetune_layout.addWidget(QLabel("Model directory:"), 1, 0)
        self.finetune_layout.addWidget(self.model_dir, 1, 1)

        self.finetune_layout.addWidget(QLabel("Finetune layers: "), 2, 0)
        self.finetune_layout.addWidget(self.finetune_layers, 2, 1)

        self.finetune_layout.addWidget(QLabel("Epochs: "), 3, 0)
        self.finetune_layout.addWidget(self.epochs, 3, 1)

        self.finetune_layout.addWidget(QLabel("save dir"), 4, 0)
        self.finetune_layout.addWidget(self.save_dir, 4, 1)

        self.finetune_layout.addWidget(QLabel("Finetuned model name: "), 5, 0)
        self.finetune_layout.addWidget(self.model_save_name, 5, 1)

        self.finetune_layout.addWidget(self.finetune_btn, 6, 0, 1, 2)

        # Adding model to model registry
        self.model_task = QLineEdit(
            placeholderText="e.g. mito"
        )  # this may need translating (there should be a translation function)
        self.manifest_name = QLineEdit(placeholderText="e.g. empanada")
        self.model_ckp_location = QLineEdit(
            placeholderText="Path to model checkpoint"
        )
        self.add_model_btn = QPushButton("add model to registry")
        # name task location, manifestname
        self.add_model_btn.clicked.connect(self.add_model_to_registry)

        self.finetune_layout.addWidget(QLabel("model task: "), 7, 0)
        self.finetune_layout.addWidget(self.model_task, 7, 1)

        self.finetune_layout.addWidget(QLabel("manifest name: "), 8, 0)
        self.finetune_layout.addWidget(self.manifest_name, 8, 1)

        self.finetune_layout.addWidget(QLabel("model checkpoint path: "), 9, 0)
        self.finetune_layout.addWidget(self.model_ckp_location, 9, 1)

        self.finetune_layout.addWidget(self.add_model_btn, 10, 0, 1, 2)

        self.inner_layout.addWidget(self.finetune_box)

    def run_finetuning(self):
        print("run_finetuning in finetune_widget")
        # collect user input of the finetuning widget and convert into a config

        finetuning_config = {}

        finetuning_config["train_dir"] = self.train_dir.text()
        finetuning_config["model_dir"] = self.model_dir.text()
        finetuning_config["save_dir"] = self.save_dir.text()
        finetuning_config["save_name"] = self.model_save_name.text()
        finetuning_config["layers"] = self.finetune_layers.currentText()
        finetuning_config["epochs"] = self.epochs.value()

        finetune(finetuning_config)

    def add_model_to_registry(self):
        print("saving model to registry...")
        model_name = self.model_save_name.text()
        model_task = self.model_task.text()
        location = self.model_ckp_location.text()

        manifest_name = self.manifest_name.text()

        add_model_local(model_name, model_task, location, manifest_name)
        print("saved model to registry - please see the inference widget")
