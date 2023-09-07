from abc import abstractmethod
from typing import Optional

import napari
from qtpy.QtWidgets import QWidget, QLayout, QGridLayout, QGroupBox


class SubWidget(QWidget):
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

    def __init__(
        self,
        viewer: napari.Viewer,
        title: str,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        super().__init__()
        self.viewer = viewer
        self.parent = parent

        # Set the layout
        self.setLayout(layout())
        # Set the main widget container
        self.widget = QGroupBox(f"{title.capitalize()}:")

    @abstractmethod
    def create_box(self):
        raise NotImplementedError
