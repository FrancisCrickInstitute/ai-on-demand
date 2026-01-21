import napari
from .run_finetuning import finetune
from typing import Optional
from pathlib import Path

from aiod_registry import add_model_local, load_manifests
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
from ai_on_demand.widget_classes import SubWidget, QGroupBox
from ai_on_demand.utils import format_tooltip


class FinetuneParameters(SubWidget):
    _name = "finetune_params"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        super().__init__(
            viewer=viewer,
            title="Finetune Parameters",
            parent=parent,
            layout=layout,
            tooltip=parent.tooltip,
        )

    default_save_location = "/Users/ahmedn/.nextflow/aiod/aiod_cache/finetuned_models/"  # extact this from the base dir stuff

    def create_box(self):
        self.finetune_box = QGroupBox("Finetune Model")

        self.finetune_layout = QGridLayout()
        self.finetune_box.setLayout(self.finetune_layout)

        self.train_dir = QLineEdit(placeholderText="Train directory")

        self.finetune_layers = QComboBox()
        self.finetune_layers.addItems(
            [
                "none",
                "layer1",
                "layer2",
                "layer3",
                "layer4",
                "all",
            ]
        )  # These layer refer to the encoder layers to unfreeze - "none" would be only decoder finetuning
        self.epochs = QSpinBox()
        self.epochs.setRange(0, 1000)
        # TODO: would we ever do finetuning for more than 1000 epochs?! maybe someone like Jon wants to retrain the model should we prevent that
        self.epochs.setValue(5)
        self.model_save_name = QLineEdit(
            placeholderText="Name you finetuned model"
        )
        # TODO: maybe not a good idea to let users pick names can be automatic like {model_name}_{finetuned}_{#}?

        self.finetune_layout.addWidget(QLabel("Train directory:"), 0, 0)
        self.finetune_layout.addWidget(self.train_dir, 0, 1)

        self.finetune_layout.addWidget(QLabel("Finetune layers: "), 2, 0)
        self.finetune_layout.addWidget(self.finetune_layers, 2, 1)

        self.finetune_layout.addWidget(QLabel("Epochs: "), 3, 0)
        self.finetune_layout.addWidget(self.epochs, 3, 1)

        self.finetune_layout.addWidget(QLabel("Finetuned model name: "), 5, 0)
        self.finetune_layout.addWidget(self.model_save_name, 5, 1)

        self.manifest_name = QLineEdit(placeholderText="e.g. empanada")
        self.add_model_btn = QPushButton("add model to registry")
        self.add_model_btn.setDisabled(True)
        self.add_model_btn.setToolTip(
            format_tooltip(
                "Adding model becomes available after running pipeline once"
            )
        )
        # name task location, manifestname
        self.add_model_btn.clicked.connect(self.add_model_to_registry)

        self.finetune_layout.addWidget(self.add_model_btn, 6, 0, 1, 2)

        self.inner_layout.addWidget(self.finetune_box)

    def enable_add_model(self, nxf_base_dir: str):
        self.nxf_base_dir = nxf_base_dir
        self.add_model_btn.setDisabled(False)

    def add_model_to_registry(self):
        print("saving model to registry...")
        model_name = self.model_save_name.text()
        model_task = self.parent.selected_task
        model_save_fpath = (
            f"{self.nxf_base_dir}/aiod_cache/finetuned_models/{model_name}.pth"
        )

        manifest_name = self.parent.selected_model

        add_model_local(
            model_name, model_task, model_save_fpath, manifest_name
        )
        print("saved model to registry - please see the inference widget")
