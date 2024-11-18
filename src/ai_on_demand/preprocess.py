from functools import partial
from typing import Optional

import napari
from napari.utils.notifications import show_error, show_info
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QVBoxLayout,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
)
import qtpy.QtCore

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import format_tooltip

import aiod_utils
from aiod_utils.preprocess import get_preprocess_methods, get_preprocess_params


class PreprocessWidget(SubWidget):
    _name = "preprocess"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QVBoxLayout,
        **kwargs,
    ):
        # Load and extract all the available preprocessing options
        self.preprocess_methods = get_preprocess_methods()
        # Store the elements for later extraction
        self.preprocess_boxes = {}
        # Store the order of the preprocessing
        self.preprocess_order = None
        # Store the order as a list for easier manipulation
        self.order_list = []

        super().__init__(
            viewer=viewer,
            title="Preprocessing",
            parent=parent,
            layout=layout,
            tooltip="""
Select image preprocessing options. Note that all preprocessing is done on-the-fly in Nextflow.

Any preprocessing applied here is for visualization purposes only, only the original image will be used in the Nextflow pipeline.
""",
            **kwargs,
        )

    def create_box(self, variant: Optional[str] = None):
        # Need to create these first as they are used in the callback
        self.order_label = QLabel("Preprocessing order:")
        self.preprocess_order = QLineEdit()
        self.preprocess_order.setReadOnly(True)
        self.init_order = "None selected!"
        self.preprocess_order.setText(self.init_order)
        # Go through each method, creating a box and populating the UI elements for each parameter
        for name, d in self.preprocess_methods.items():
            group_box = QGroupBox(name)
            self.preprocess_boxes[name] = {
                "box": group_box,
                "params": {},
            }
            if getattr(d["object"], "tooltip", None) is not None:
                group_box.setToolTip(format_tooltip(d["object"].tooltip))
            group_box.setCheckable(True)
            group_box.setChecked(False)
            group_box.toggled.connect(self.on_click_preprocess(name))
            group_layout = QGridLayout()
            group_box.setLayout(group_layout)

            # Loop through params
            for i, (param_name, param_dict) in enumerate(d["params"].items()):
                # Create the label
                label = QLabel(param_dict["name"])
                group_layout.addWidget(label, i, 0, 1, 1)

                # Create the input based on type
                # If values key exist, multiple options to select
                if "values" in param_dict:
                    widget = QComboBox()
                    for value in param_dict["values"]:
                        widget.addItem(value)
                    # Set the default value
                    widget.setCurrentIndex(
                        param_dict["values"].index(param_dict["default"])
                    )
                elif isinstance(param_dict["default"], bool):
                    widget = QCheckBox()
                    if param_dict["default"]:
                        widget.setChecked(True)
                    else:
                        widget.setChecked(False)
                elif isinstance(param_dict["default"], (int, float, str)):
                    widget = QLineEdit()
                    widget.setText(str(param_dict["default"]))
                # Get cleaner representation of list/tuple (avoid () & [])
                elif isinstance(param_dict["default"], (list, tuple)):
                    widget = QLineEdit()
                    widget.setText(", ".join(map(str, param_dict["default"])))
                else:
                    raise ValueError(
                        f"Parameter {param_name} of preprocess method {name} has an invalid type ({type(param_dict['default'])})."
                    )
                # Add tooltip if available
                if "tooltip" in param_dict:
                    label.setToolTip(format_tooltip(param_dict["tooltip"]))
                    widget.setToolTip(format_tooltip(param_dict["tooltip"]))
                # Add the widget to the layout
                group_layout.addWidget(widget, i, 1, 1, 1)
                # Store the widget to extract the value of later
                self.preprocess_boxes[name]["params"][param_name] = widget

            # Add the group box to the inner layout
            self.inner_layout.addWidget(group_box)
        # Create a layout for the order and preview btns
        self.btn_widget = QWidget()
        self.btn_layout = QGridLayout()
        self.btn_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)
        # Add text box to show current order of preprocessing
        self.btn_layout.addWidget(self.order_label, 0, 0, 1, 1)
        self.btn_layout.addWidget(self.preprocess_order, 0, 1, 1, 2)
        # Add preview button
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(
            partial(self.on_click_run, preview=True)
        )
        self.preview_btn.setToolTip(
            format_tooltip(
                "Preview the effect of the selected preprocessing options on the currently selected image (or first image layer if none selected)."
            )
        )
        self.btn_layout.addWidget(self.preview_btn, 1, 0, 1, 1)
        # Add a run button to apply the preprocessing entirely
        self.prep_run_btn = QPushButton("Run")
        self.prep_run_btn.clicked.connect(
            partial(self.on_click_run, preview=False)
        )
        self.prep_run_btn.setToolTip(
            format_tooltip(
                """
Apply the selected preprocessing options to the currently selected image (or first image layer if none selected).
NOTE: This will run the computation locally and return an array in-memory, so be careful with larger images and/or expensive preprocessing.
NOTE: The result is just for visualization, and will not be used in the Nextflow pipeline.
                """
            )
        )
        self.btn_layout.addWidget(self.prep_run_btn, 1, 1, 1, 1)
        # Add a button for rescaling images
        # Best for visually comparing downsampling with masks
        self.rescale_btn = QPushButton("Rescale masks")
        self.rescale_btn.clicked.connect(self.on_click_rescale)
        self.rescale_btn.setToolTip(
            format_tooltip(
                """
Rescale mask layers to raw data size (if downsampled). Helps visually compare with the original data.
                """
            )
        )
        self.btn_layout.addWidget(self.rescale_btn, 1, 2, 1, 1)
        # Set the layout for the widget
        self.btn_widget.setLayout(self.btn_layout)
        self.inner_layout.addWidget(self.btn_widget)
        self.inner_widget.setLayout(self.inner_layout)

    def on_click_preprocess(self, name: str):
        # Callback for when a preprocess method is selected
        def cb():
            # Get the box to check if it is checked
            group_box = self.preprocess_boxes[name]["box"]
            checked = group_box.isChecked()
            order = self.preprocess_order.text()
            if order == self.init_order:
                order = name
                self.order_list = [name]
            else:
                self.order_list = order.split("->")
                # If checked, add to the start of the list
                if checked:
                    self.order_list.append(name)
                else:
                    self.order_list.remove(name)
                # Handle when all are unchecked
                if len(self.order_list) == 0:
                    order = self.init_order
                else:
                    order = "->".join(self.order_list)
            self.preprocess_order.setText(order)

        # Return the callback
        return cb

    def on_click_run(self, preview: bool = False):
        # Callback for when the preview button is clicked
        # First check if we are able to preview
        if self.preprocess_order.text() == self.init_order:
            show_error(
                "No preprocessing methods selected! Please select at least one preprocessing method to preview.",
            )
            return
        if len(self.viewer.layers) == 0:
            show_error(
                "No image layers available! Please load an image layer to preview the preprocessing effect on.",
            )
            return
        # Extract the options from the UI elements
        options = self.extract_options()
        # Get the selected image
        if isinstance(
            self.viewer.layers.selection.active, napari.layers.Image
        ):
            layer = self.viewer.layers.selection.active
        else:
            # NOTE: Use -1 as that's top of the list?
            layer = [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Image)
            ][0]
        # Extract just the slice of the data
        data = layer.data
        if data.ndim == 3:
            # Get the current slice
            if preview:
                image = data[self.viewer.dims.current_step[0]]
                # As the preview is for 2D only, remap 3D-specific options to 2D if needed
                for option in options:
                    if option["name"] == "Filter":
                        footprint = option["params"]["footprint"]
                        if footprint == "cube":
                            option["params"]["footprint"] = "square"
                        elif footprint == "ball":
                            option["params"]["footprint"] = "disk"
                        # Show info if changed
                        if footprint != option["params"]["footprint"]:
                            show_info(
                                f"Changed Filter footprint to {option['params']['footprint']} from {footprint} for 2D preview."
                            )
                    elif option["name"] == "Downsample":
                        blocksize = option["params"]["block_size"]
                        if len(blocksize) == 3:
                            option["params"]["block_size"] = blocksize[1:]
                            show_info(
                                f"Changed Downsample blocksize to {option['params']['block_size']} from {blocksize} for 2D preview."
                            )
            else:
                image = data
        else:
            # Convert to numpy?
            image = data
        # Extract blocksize for rescaling if downsampling used
        # This will be the corrected blocksize based on preview/run and input data shape
        blocksize = aiod_utils.preprocess.get_downsample_factor(options)
        # Apply the preprocessing and show the result
        # Convert to numpy array in case it's dask
        image = aiod_utils.run_preprocess(np.array(image), options)
        prep_str = get_preprocess_params(options)
        # Add metadata to skip file path checks in plugin
        self.viewer.add_image(
            data=image,
            name=f"{layer.name}_{prep_str}",
            metadata={
                "preprocess": True,
                "downsample_blocksize": blocksize,
            },
        )
        # Switch focus back to the original layer
        self.viewer.layers.selection.active = layer

    def on_click_rescale(self):
        # Gather all the layers on which preprocessing has been applied
        mask_layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]
        if len(mask_layers) == 0:
            show_error(
                "No mask layers found to rescale!",
            )
            return
        for layer in mask_layers:
            blocksize = layer.metadata.get("downsample_factor", None)
            if blocksize is not None:
                layer.scale = blocksize

    def extract_options(self):
        # Extract the options from the UI elements
        options = []
        # Loop over the specified order
        for name in self.order_list:
            # Get the method and the widget dict
            method_dict = self.preprocess_methods[name]
            widget_dict = self.preprocess_boxes[name]
            # Create the sub-dict for the method
            option_dict = {"name": name, "params": {}}
            for param_name, widget in widget_dict["params"].items():
                # Get the default dtype to cast the value back
                dtype = type(method_dict["params"][param_name]["default"])
                # Extract the value based on the widget type
                if isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    value = widget.text()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                # Some elements converts all to str, so cast back just in case
                if dtype in (list, tuple):
                    # Get internal dtype to cast individual elements
                    internal_dtype = type(
                        method_dict["params"][param_name]["default"][0]
                    )
                    # NOTE: We always cast to list to avoid '!!python/tuple' pyyaml tag
                    # As this cannot be loaded by the yaml.safe_load function
                    # And I do not want to use another loader for configs that users can write
                    option_dict["params"][param_name] = list(
                        map(internal_dtype, value.replace(" ", "").split(","))
                    )
                else:
                    option_dict["params"][param_name] = dtype(value)
            # Add the method dict to the options list
            options.append(option_dict)
        return options
