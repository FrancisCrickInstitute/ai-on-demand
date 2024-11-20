from abc import abstractmethod
from pathlib import Path
import string
from typing import Optional
import yaml

import napari
from npe2 import PluginManager
from qtpy.QtWidgets import (
    QWidget,
    QScrollArea,
    QLayout,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QFrame,
)
from qtpy.QtGui import QPixmap
import qtpy.QtCore

from ai_on_demand.qcollapsible import QCollapsible
from ai_on_demand.utils import (
    format_tooltip,
    get_plugin_cache,
)


class MainWidget(QWidget):
    def __init__(
        self,
        napari_viewer: napari.Viewer,
        title: str,
        tooltip: Optional[str] = None,
    ):
        super().__init__()
        pm = PluginManager.instance()
        self.all_manifests = pm.commands.execute("ai-on-demand.get_manifests")
        self.plugin_settings = pm.commands.execute("ai-on-demand.get_settings")

        self.viewer = napari_viewer
        self.scroll = QScrollArea()

        # Set overall layout for the widget
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(qtpy.QtCore.Qt.AlignTop)

        # Dictionary to contain all subwidgets
        self.subwidgets = {}

        # A hash to uniquely identify a run
        # Only used to uniquely identify a Nextflow pipeline based on inputs
        self.run_hash = None

        # Add a Crick logo to the widget
        self.logo_label = QLabel()
        logo = QPixmap(
            str(
                Path(__file__).parent
                / "resources"
                / "CRICK_Brandmark_01_transparent.png"
            )
        ).scaledToHeight(100, mode=qtpy.QtCore.Qt.SmoothTransformation)
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
            self.tooltip = tooltip
            self.title.setToolTip(format_tooltip(tooltip))
        self.layout().addWidget(self.title)

        # Create the widget that will be used to add subwidgets to
        # This is then the widget for the scroll area, to the logo/title is excluded from scrolling
        self.content_widget = QWidget()
        self.content_widget.setLayout(QVBoxLayout())
        self.scroll.setWidgetResizable(True)
        # This is needed to avoid unnecessary spacing when in the ScrollArea
        self.content_widget.setSizePolicy(
            qtpy.QtWidgets.QSizePolicy.Minimum,
            qtpy.QtWidgets.QSizePolicy.Fixed,
        )
        self.content_widget.layout().setAlignment(qtpy.QtCore.Qt.AlignTop)
        self.scroll.setWidget(self.content_widget)
        self.layout().addWidget(self.scroll)

    def register_widget(self, widget: "SubWidget"):
        self.subwidgets[widget._name] = widget

    def store_settings(self):
        # Check for each of the things we want to store
        # Skipping if not present in this main widget
        if "nxf" in self.subwidgets:
            self.plugin_settings["nxf"] = self.subwidgets["nxf"].get_settings()
        # TODO: Think/check if we want to store anything else
        # Save the settings to the cache
        # As we retrieve everything every time, we can just overwrite the file
        _, settings_path = get_plugin_cache()
        with open(settings_path, "w") as f:
            yaml.dump(self.plugin_settings, f)

    @abstractmethod
    def get_run_hash(self):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        raise NotImplementedError


class SubWidget(QCollapsible):
    # Define a shorthand name to be used to register the widget
    _name: str = None

    def __init__(
        self,
        viewer: napari.Viewer,
        title: str,
        parent: Optional[QWidget] = None,
        layout: QLayout = QVBoxLayout,
        tooltip: Optional[str] = None,
        **kwargs,
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
            Layout to use for the widget. This is the default layout for the subwidget.
        tooltip : Optional[str], optional
            Tooltip to display for the widget, by default None.
        kwargs
            Additional keyword arguments to pass to the QCollapsible widget, such as margins, animation duration, etc.
        """
        super().__init__(
            title=string.capwords(title),
            layout=layout,
            collapsedIcon="▶",
            expandedIcon="▼",
            duration=200,
            margins=(5, 0, 5, 0),
            **kwargs,
        )
        self.viewer = viewer
        self.parent = parent
        self.title = title

        # Set the inner widgets (the things that get collapsed/expanded)
        self.inner_widget = QWidget()
        self.inner_layout = QGridLayout()
        self.inner_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)

        # Set the tooltip if given
        if tooltip is not None:
            self.setToolTip(format_tooltip(tooltip))
        # Create the initial widgets/elements
        self.create_box()
        self.inner_widget.setLayout(self.inner_layout)
        # Add the inner widget to the collapsible widget
        self.addWidget(self.inner_widget)
        # Add a divider line to better separate subwidgets
        # NOTE: Currently invisible, but just a spacer
        # btn_colour = self._toggle_btn.palette().button().color().name()  # Tries to get the button colour
        divider_line = QFrame()
        divider_line.setFrameShape(QFrame.HLine)
        # divider_line.setFrameShadow(QFrame.Sunken)
        divider_line.setStyleSheet(
            """
            QFrame[frameShape='4'] {
                border: none;
            }
        """
        )
        # Ensure minimal space taken
        divider_line.setMaximumHeight(1)
        self.content().layout().addWidget(divider_line)

        # If given a parent at creation, add this widget to the parent's layout
        if self.parent is not None:
            # Add to the content widget (i.e. scrollable area)
            self.parent.content_widget.layout().addWidget(self)

        # Load any previous settings for this widget if available
        self.load_settings()

    @abstractmethod
    def create_box(self, variant: Optional[str] = None):
        """
        Create the box for the subwidget, i.e. all UI elements.
        """
        raise NotImplementedError

    @abstractmethod
    def load_settings(self):
        """
        Load settings for the subwidget.
        """
        pass

    @abstractmethod
    def get_settings(self):
        """
        Get settings for the subwidget.
        """
        pass
