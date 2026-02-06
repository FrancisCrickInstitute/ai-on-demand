import napari
from typing import Optional
from pathlib import Path

from napari.utils.notifications import show_info
from aiod_registry import add_model_local, load_manifests
from napari._qt.qt_resources import QColoredSVGIcon
from qtpy.QtWidgets import (
    QWidget,
    QGridLayout,
    QLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QFileDialog,
    QMessageBox,
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

        self.finetuning_meta_data = None

    def create_box(self):
        self.finetune_box = QGroupBox("Finetune Model")

        self.finetune_layout = QGridLayout()
        self.finetune_box.setLayout(self.finetune_layout)

        self.train_dir = QLineEdit(placeholderText="Train directory")

        self.finetune_layers = QComboBox()

        self.patch_size = QLineEdit(placeholderText="Height,Width")
        self.patch_size.setText("64,64")
        self.epochs = QSpinBox()
        self.epochs.setRange(0, 1000)
        # TODO: would we ever do finetuning for more than 1000 epochs?! maybe someone like Jon wants to retrain the model should we prevent that
        self.epochs.setValue(5)
        self.model_save_name = QLineEdit(
            placeholderText="Name you finetuned model"
        )
        # TODO: maybe not a good idea to let users pick names can be automatic like {model_name}_{finetuned}_{#}?

        self.train_dir_label = QLabel("train directory:")
        train_dir_tooltip = "Select the directory where you have saved the training data with /images, /labels"
        self.train_dir_label.setToolTip(format_tooltip(train_dir_tooltip))
        self.train_dir_text = QLabel("")
        self.train_dir_text.setWordWrap(True)
        self.train_dir_text.setToolTip(
            format_tooltip("The selected train directory.")
        )
        self.train_dir_text.setMaximumWidth(400)
        # Button to change the base directory
        self.train_dir_btn = QPushButton("Locate Training Data")
        self.train_dir_btn.clicked.connect(self.on_click_train_dir)
        self.train_dir_btn.setToolTip(format_tooltip(train_dir_tooltip))
        self.train_dir_info = QPushButton("")
        self.train_dir_info.setIcon(
            QColoredSVGIcon.from_resources("help").colored(theme="dark")
        )
        self.train_dir_info.setFixedWidth(30)
        self.train_dir_info.setToolTip("Help I don't how to structure my data")
        self.train_dir_info.clicked.connect(self._show_train_dir_info)

        self.finetune_layout.addWidget(self.train_dir_label, 0, 0)
        self.finetune_layout.addWidget(self.train_dir_text, 0, 1, 1, 2)
        self.finetune_layout.addWidget(self.train_dir_btn, 1, 0, 1, 2)
        self.finetune_layout.addWidget(self.train_dir_info, 1, 2)

        self.finetune_layout.addWidget(QLabel("Patch size"), 2, 0)
        self.finetune_layout.addWidget(self.patch_size, 2, 1, 1, 2)

        self.finetune_layout.addWidget(QLabel("Finetune layers: "), 3, 0)
        self.finetune_layout.addWidget(self.finetune_layers, 3, 1, 1, 2)

        self.finetune_layout.addWidget(QLabel("Epochs: "), 4, 0)
        self.finetune_layout.addWidget(self.epochs, 4, 1, 1, 2)

        self.finetune_layout.addWidget(QLabel("Finetuned model name: "), 5, 0)
        self.finetune_layout.addWidget(self.model_save_name, 5, 1, 1, 2)

        self.manifest_name = QLineEdit(placeholderText="e.g. empanada")
        self.add_model_btn = QPushButton("Add Model To Registry")
        self.add_model_btn.setDisabled(True)
        self.add_model_btn.setToolTip(
            format_tooltip(
                "Adding model becomes available after running pipeline once"
            )
        )
        # name task location, manifestname
        self.add_model_btn.clicked.connect(self.add_model_to_registry)

        self.finetune_layout.addWidget(self.add_model_btn, 6, 0, 1, 3)

        self.inner_layout.addWidget(self.finetune_box)

    def on_click_train_dir(self):
        """
        Callback for when the train directory button is clicked. Opens a dialog to select a directory to get the trianing data from.
        """
        train_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select directory where the training data is",
            directory=None,
        )
        # Skip if no directory selected
        if train_dir == "":
            return
        # Replace any spaces, makes everything else easier
        new_dir_name = Path(train_dir).name.replace(" ", "_")
        train_dir = Path(train_dir).parent / new_dir_name
        # Update the text
        self.train_dir_text.setText(str(train_dir))

    def _show_train_dir_info(self):
        QMessageBox.information(
            self,
            "Training Data Information",
            (
                "Training data should be oranised in to 1 single directory containing:\n"
                "images/ and masks/:\n"
                "images and masks are paired by\n"  # TODO: make a clear folder structure and instructions
            ),
        )

    def update_finetune_layers(self, task_model_verson):
        self.finetune_layers.clear()
        version_data = self.parent.subwidgets["model"].model_version_tasks[
            task_model_verson
        ]
        # save for later use when saving the model
        self.finetuning_meta_data = version_data.finetuning_meta_data
        avail_layers = self.finetuning_meta_data.avail_layers
        self.finetune_layers.addItems(avail_layers)

    def enable_add_model(self, nxf_base_dir: str):
        self.nxf_base_dir = nxf_base_dir
        self.add_model_btn.setDisabled(False)

    def add_model_to_registry(self):
        print("saving model to registry...")
        model_name = self.model_save_name.text()
        model_task = self.parent.selected_task
        model_save_fpath = (
            f"{self.nxf_base_dir}/aiod_cache/finetune_cache/{model_name}.pth"
        )
        manifest_name = self.parent.selected_model

        add_model_local(
            model_name,
            model_task,
            model_save_fpath,
            manifest_name,
            dict(self.finetuning_meta_data),
        )

        self.parent.refresh_instances(
            instances_to_refresh=["Inference", "Finetuning"]
        )

        show_info(
            "Fine-tuned model has been saved to registry and is ready to use"
        )
