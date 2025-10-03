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
    QGroupBox,
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
        # Extract settings for every subwidget that has implemented the get_settings method
        for k, subwidget in self.subwidgets.items():
            settings = subwidget.get_settings()
            if settings is not None:
                self.plugin_settings[k] = settings
        # TODO: Think/check if we want to store anything else
        # Save the settings to the cache
        # As we retrieve everything every time, we can just overwrite the file
        _, settings_path = get_plugin_cache()
        print(" - this is the settings variable - ")
        print(self.plugin_settings)
        with open(settings_path, "w") as f:
            print(f'this is where the settings are: ',settings_path)
            yaml.dump(self.plugin_settings, f)

    def store_config(self):
        config_name = self.subwidgets['nxf'].config_name.text().strip()
        if not config_name:
            config_name = "aiod-config" # default config name
        
        # get next flow config settings from the pipeline param
        nxfWidget = self.subwidgets.get('nxf')
        nxf_cmd, nxf_params, proceed, img_paths = nxfWidget.pipelines[
                nxfWidget.pipeline
            ]["setup"]()
        
        
        # Extract settings for every subwidget that has implemented the get_settings method
        config_settings = nxf_params
        for k, subwidget in self.subwidgets.items():
            settings = subwidget.get_settings()
            if settings is not None:
                config_settings[k] = settings
        
        # If a unique file name is given by the user
        cache_dir, _ = get_plugin_cache()
        config_file_path = cache_dir / f"{config_name}.yaml"
        print("-- config file location: ", config_file_path)
        
        # Save the config to its own file
        with open(config_file_path, "w") as f: # this will over write if the file already exists*
            print('writing to config')
            yaml.dump(config_settings, f)
        
        print(f"Config saved as: {config_file_path}")
    
    def load_config(self, config_path):
        """Load a specific config and apply to all subwidgets"""
        with open(config_path, "r") as f:
            print('was able to open')
            config_data = yaml.safe_load(f)
        print(config_data)

        print(" -- these are the availbe subwidgets -- ")
        print(self.subwidgets)

        # -- Load Segmentation Task config -- 
        config_task = config_data['task']
        print("loading config...")
        task_widget = self.subwidgets.get("task")
        
        # Uncheck all buttons first
        for btn in task_widget.task_buttons.values():
            btn.setChecked(False)
        
        # Check the correct task button
        task_widget.task_buttons[config_task].setChecked(True)
        
        # Trigger the callback to update other widgets
        task_widget.on_click_task()

        # -- Load Model Selection config --
        config_model = config_data['model']
        config_model_version = config_data['model_type']
        model_widget = self.subwidgets.get('model') 

        if config_model and model_widget:
            # First set the model dropdown
            model_display_name = model_widget.base_to_display.get(config_model)
            if model_display_name:
                model_index = model_widget.model_dropdown.findText(model_display_name)
                if model_index != -1:
                    model_widget.model_dropdown.setCurrentIndex(model_index)
                    # Trigger the model selection callback to populate versions
                    model_widget.on_model_select()
                    
                    print('trying to find model version: ', config_model_version)
                    model_version_index = model_widget.model_version_dropdown.findText(config_model_version)

                    print(' -- These are the available versions -- ')
                    available_versions = [model_widget.model_version_dropdown.itemText(i) 
                                          for i in range(model_widget.model_version_dropdown.count())]
                    print(available_versions)

                    if model_version_index != -1:
                        model_widget.model_version_dropdown.setCurrentIndex(model_version_index)
                        # Trigger the version selection callback
                        model_widget.on_model_version_select()
                        print(f"Set model version to: {config_model_version}")
                    else:
                        print('Model version not found')
                else:
                    print(f'Model {config_model} not found')
            else:
                print(f'Model display name not found for {config_model}')
        
        # -- Load Preprocessing from config -- 
        config_preprocess = config_data['preprocess']
        preprocess_widget = self.subwidgets.get('preprocess')


        if config_preprocess and preprocess_widget:
            print ('config preprocess: ',config_preprocess)
            print ('config 1 in preprocess: ',config_preprocess[0])
            # Check for 'Filter' in preprocessing steps
            for step in config_preprocess:
                method_name = step['name']
                method_params = step['params']

                print('adding from config method: ', method_name, method_params)

                if method_name in preprocess_widget.preprocess_boxes:
                    group_box = preprocess_widget.preprocess_boxes[method_name]['box']
                    group_box.setChecked(True)
                
                # for param_name, param_values in method_params.items():
                        # if hasattr(widget, 'setChecked'):  # QCheckBox
                        #     widget.setChecked(bool(param_value))
                        # elif hasattr(widget, 'setText'):  # QLineEdit
                        #     if isinstance(param_value, (list, tuple)):
                        #         widget.setText(", ".join(map(str, param_value)))
                        #     else:
                        #         widget.setText(str(param_value))
                        # elif hasattr(widget, 'setCurrentText'):  # QComboBox
                        #     widget.setCurrentText(str(param_value))
                        # else:
                        #     print(f"Unknown widget type for {param_name}")
                    
                
            # for name, d in preprocess_widget.preprocess_methods.items():
            #     print('type for name in preprocess method item - ', type(name))
            #     print('type for d in preprocess method item - ', type(d))
            #     if config_preprocess[name]:


        # -- Load nextflow config -- 
        config_nxf = config_data['nxf']
        config_profile = config_nxf['profile'] 
        
        print(' ---- these are the subwidgets: ', self.subwidgets)
        nxf_widget = self.subwidgets.get('nxf')
        
        # Set the execution profile in the combo box
        if nxf_widget and config_profile:
            profile_index = nxf_widget.nxf_profile_box.findText(config_profile)
            
            if profile_index != -1:
                nxf_widget.nxf_profile_box.setCurrentIndex(profile_index)
                print(f"Set profile to: {config_profile}")
            else:
                print(f"Profile '{config_profile}' not found in available profiles")
                available_profiles = [nxf_widget.nxf_profile_box.itemText(i) 
                                    for i in range(nxf_widget.nxf_profile_box.count())]
                print(f"Available profiles: {available_profiles}")
        

    @abstractmethod
    def get_run_hash(self):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        raise NotImplementedError

    def store_widget_settings(self):
        """
        Store the settings for the widget.
        """
        pass

    def store_subwidget_settings(self):
        """
        Store the settings for the subwidgets.
        """
        for widget in self.subwidgets.values():
            widget.store_settings()

    def load_config_file(self, config: dict):
        """
        Load a config file for the widget.
        """
        # Subwidget names: ['task', 'model', 'data', 'preprocess', 'nxf', 'config', 'export']
        for subwidget in self.subwidgets.values():
            if subwidget._name in config:
                print(' -- subwidget name: ',subwidget._name)
                print(' -- type of config: ', type(config[subwidget._name]))
                subwidget.load_config(config=config[subwidget._name])


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

        if kwargs.get("expanded", False):
            self.expand(animate=False)

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

    @abstractmethod
    def load_config(self, config: dict):
        """
        Load a specific config and apply to the subwidget.
        """
        pass

    def _make_separator(self):
        """
        Create a thin separator line to better separate elements within a subwidget.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Raised)
        separator.setSizePolicy(
            qtpy.QtWidgets.QSizePolicy.Expanding,
            qtpy.QtWidgets.QSizePolicy.Minimum,
        )
        colour = napari.utils.theme.get_theme("dark").secondary.as_rgb()
        separator.setStyleSheet(
            f"border: 1px solid {colour}; background-color: {colour};"
        )
        return separator

    def _make_groupbox(self, title: str, tooltip: Optional[str] = None):
        group_box = QGroupBox(title)
        if tooltip is not None:
            group_box.setToolTip(format_tooltip(tooltip))
        group_box.setCheckable(False)
        group_layout = QGridLayout()
        group_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)
        group_box.setLayout(group_layout)
        group_box.setContentsMargins(0, 0, 0, 0)
        return group_box
