from pathlib import Path
from typing import Union
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel, QLineEdit, QRadioButton, QGroupBox
from qtpy.QtGui import QPixmap
import qtpy.QtCore

import napari


class AIOnDemand(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.layout().addWidget(btn)

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
