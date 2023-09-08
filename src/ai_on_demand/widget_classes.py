from abc import abstractmethod
from pathlib import Path
from typing import Optional

import napari
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)
from qtpy.QtGui import QPixmap
import qtpy.QtCore

from ai_on_demand.utils import format_tooltip


class MainWidget(QWidget):
    def __init__(
        self,
        napari_viewer: napari.Viewer,
        title: str,
        tooltip: Optional[str] = None,
    ):
        super().__init__()
        self.viewer = napari_viewer

        # Set overall layout for the widget
        self.setLayout(QVBoxLayout())

        # Dictionary to contain all subwidgets
        self.subwidgets = {}

        # Add a Crick logo to the widget
        self.logo_label = QLabel()
        logo = QPixmap(
            str(
                Path(__file__).parent
                / "resources"
                / "CRICK_Brandmark_01_transparent.png"
            )
        ).scaledToHeight(125, mode=qtpy.QtCore.Qt.SmoothTransformation)
        self.logo_label.setPixmap(logo)
        self.logo_label.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.layout().addWidget(self.logo_label)

        # Widget title to display
        self.title = QLabel(f"AI OnDemand: {title}")
        title_font = self.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.title.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        if tooltip is not None:
            self.title.setToolTip(format_tooltip(tooltip))
        # self.title.adjustSize()
        self.layout().addWidget(self.title)

    def register_widget(self, widget: "SubWidget"):
        self.subwidgets[widget._name] = widget


class SubWidget(QWidget):
    # Define a shorthand name to be used to register the widget
    _name: str = None

    def __init__(
        self,
        viewer: napari.Viewer,
        title: str,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        """
        Custom widget for the AI OnDemand plugin.

        Controls the subwidgets/modules of the plugin which are used for different meta-plugins.
        Allows for easy changes of style, uniform layout, and better interoperability between other subwidgets.

        Parameters
        ----------
        viewer : napari.Viewer
            Napari viewer object.
        parent : QWidget, optional
            Parent widget, by default None. Allows for easy access to the parent widget and its attributes.
        title : str
            Title of the widget to be displayed.
        layout : QLayout, optional
            Layout to use for the widget, by default QGridLayout. This is the default layout for the subwidget.
        """
        super().__init__()
        self.viewer = viewer
        self.parent = parent

        # Set the layout
        self.setLayout(layout())
        # Set the main widget container
        self.widget = QGroupBox(f"{title.capitalize()}:")

        # Create the initial widgets/elements
        self.create_box()

        # If given a parent at creation, add this widget to the parent's layout
        if self.parent is not None:
            self.parent.layout().addWidget(self.widget)

    @abstractmethod
    def create_box(self, variant: Optional[str] = None):
        raise NotImplementedError
