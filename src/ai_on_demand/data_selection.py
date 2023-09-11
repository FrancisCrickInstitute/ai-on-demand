from collections import Counter
from pathlib import Path
from typing import Optional, Union

import napari
from napari.layers import Image
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QFileDialog,
)
import skimage.io

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import format_tooltip


class DataWidget(SubWidget):
    _name = "data"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        super().__init__(
            viewer,
            "data",
            parent,
            layout,
            tooltip="""
Select a directory to take images from, or select individual images.

Images can also be opened, or dragged into napari as normal. The selection will be updated accordingly. Note that all images loaded are additive, unless removed as a layer. The 'Reset selection' button can be used to clear all images.
""",
        )

        # Connect to the viewer to some callbacks
        self.viewer.layers.events.inserted.connect(self.on_layer_added)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)

    def create_box(self, variant: Optional[str] = None):
        # Create empty counter to show image load progress
        self.load_img_counter = 0
        # Create container for image paths
        self.image_path_dict = {}
        # Do a quick check to see if the user has added any images already
        if self.viewer.layers:
            for img_layer in self.viewer.layers:
                if isinstance(img_layer, Image):
                    try:
                        img_path = Path(img_layer.source.path)
                        self.image_path_dict[img_path.stem] = img_path
                    except AttributeError:
                        continue
        # Create a button to select individual images from
        self.img_btn = QPushButton("Select image files")
        self.img_btn.clicked.connect(self.browse_imgs_files)
        self.img_btn.setToolTip(
            format_tooltip(
                "Select individual image files to use as input to the model."
            )
        )
        self.layout().addWidget(self.img_btn, 0, 0)
        # TODO: What happens if multiple directories are selected? Is this possible?
        # Create a button to navigate to a directory to take images from
        self.dir_btn = QPushButton("Select image directory")
        self.dir_btn.clicked.connect(self.browse_imgs_dir)
        self.dir_btn.setToolTip(
            format_tooltip(
                """
            Select folder/directory of images to use as input to the model.
            Note that this allows for running the pipeline without having to load images into napari first.
            Any images loaded into napari will also be used within the pipeline, however.
        """
            )
        )
        self.layout().addWidget(self.dir_btn, 0, 1)
        # Add an output to show the counts
        self.init_file_msg = "No files selected or added to Napari."
        self.img_counts = QLabel(self.init_file_msg)
        self.img_counts.setWordWrap(True)
        self.layout().addWidget(self.img_counts, 1, 0, 1, 2)

        # Add a button for viewing the images within napari
        # Optional as potentially taxing, and not necessary
        self.init_view_msg = "View selected images"
        self.view_img_btn = QPushButton(self.init_view_msg)
        self.view_img_btn.setToolTip(
            format_tooltip("Load selected images into napari to view.")
        )
        self.view_img_btn.clicked.connect(self.view_images)
        self.layout().addWidget(self.view_img_btn, 2, 0)
        # Create a button to clear selected directory
        self.clear_dir_btn = QPushButton("Reset selection")
        self.clear_dir_btn.clicked.connect(self.clear_directory)
        self.clear_dir_btn.setToolTip(
            format_tooltip(
                "Reset selection of images (clears all images in the viewer)."
            )
        )
        self.layout().addWidget(self.clear_dir_btn, 2, 1)
        # Add button layout to box layout
        # Sort out layout and add to main widget
        self.widget.setLayout(self.layout())

    def on_layer_added(self, event):
        """
        Triggered whenever there is a new layer added to the viewer.

        Checks if the layer is an image, and if so, adds it to the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = event.value.source.path
            # Insert into the overall dict of images and their paths (if path is present)
            # This will be None when we are viewing arrays loaded separately from napari
            if img_path is not None:
                self.image_path_dict[Path(img_path).stem] = Path(img_path)
            # Then update the counts of files (and their types) with the extra image
            self.update_file_count()

    def on_layer_removed(self, event):
        """
        Triggered whenever a layer is removed from the viewer.

        Checks if the layer is an image, and if so, removes it from the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = event.value.source.path
            # Remove from the list of images
            if (
                img_path is not None
                and Path(img_path).stem in self.image_path_dict
            ):
                del self.image_path_dict[Path(img_path).stem]
            # Update file count with image removed
            self.update_file_count()

    def browse_imgs_files(self):
        """
        Opens a dialog for selecting images to segment.
        """
        # TODO: Implement a cache that stores the last directory used?
        # TODO: Should this cache persist across sessions, or is that invasive?
        fnames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more images",
            str(Path.home()),
            "",
        )
        if fnames != []:
            self.update_file_count(paths=fnames)

    def browse_imgs_dir(self):
        """
        Opens a dialog for selecting a directory that contains images to segment.
        """
        # TODO: Load multiple directories - https://stackoverflow.com/a/28548773/9963224
        # Quite the pain, and potentially brittle if Qt backend changes
        result = QFileDialog.getExistingDirectory(
            self, caption="Select image directory", directory=None
        )
        if result != "":
            self.update_file_count(paths=list(Path(result).glob("*")))

    def view_images(self):
        """
        Loads the selected images into napari for viewing (in separate threads).
        """
        # Return if there's nothing to show
        if len(self.image_path_dict) == 0:
            return
        # Check if there are images to load that haven't been already
        viewer_imgs = [
            Path(i.name).stem
            for i in self.viewer.layers
            if isinstance(i, Image)
        ]
        imgs_to_load = [
            v for k, v in self.image_path_dict.items() if k not in viewer_imgs
        ]
        if imgs_to_load == []:
            return
        self.view_img_btn.setEnabled(False)
        # Reset counter
        self.load_img_counter = 0

        # Create separate thread worker to avoid blocking
        @thread_worker(
            connect={
                "returned": self._add_image,
                "finished": self._reset_view_btn,
            }
        )
        def _load_image(fpath):
            return skimage.io.imread(fpath), fpath

        # Load each image in a separate thread
        for fpath in imgs_to_load:
            _load_image(fpath)
        # NOTE: This does not work well for a directory of large images on a remote directory
        # But that would trigger loading GBs into memory over network, which is...undesirable
        self.view_img_btn.setText("Loading...")

    def _add_image(self, res):
        """
        Adds an image to the viewer when loaded, using its filepath as the name.
        """
        img, fpath = res
        # Add the image to the overall dict
        self.image_path_dict[fpath.stem] = fpath
        self.viewer.add_image(img, name=fpath.stem)
        self.load_img_counter += 1
        self.view_img_btn.setText(
            f"Loading...({self.load_img_counter} image{'s' if self.load_img_counter > 1 else ''} loaded)."
        )
        img_layers = [i for i in self.viewer.layers if isinstance(i, Image)]
        # Only change text when we have as many image layers as images
        if len(img_layers) == len(self.image_path_dict):
            self.view_img_btn.setText("All images loaded.")
        # # Update the progress bar range (just in case the image wasn't loaded in time)
        # if img.ndim > 2:
        #     self.progress_bar_dict[fpath.stem].setRange(0, img.shape[-3])
        #     self.progress_bar_dict[fpath.stem].setValue(0)

    def _reset_view_btn(self):
        """Reset the view button to be clickable again when done."""
        self.view_img_btn.setEnabled(True)
        self.view_img_btn.setText("All images loaded.")
        # Also reset the viewer itself
        self.viewer.reset_view()

    def update_file_count(
        self, paths: Optional[list[Union[str, Path]]] = None
    ):
        """
        Identify all the files in a given path, and return a count
        (broken down by extension)
        """
        # Reinitialise text
        txt = "Selected "
        # Add paths to the overall list if specific ones need adding
        if paths is not None:
            for img_path in paths:
                img_path = Path(img_path)
                self.image_path_dict[img_path.stem] = img_path
        # If no files remaining, reset message and return
        if len(self.image_path_dict) == 0:
            self.img_counts.setText(self.init_file_msg)
            return
        # Get all the extensions in the path
        extension_counts = Counter(
            [i.suffix for i in self.image_path_dict.values()]
        )
        # Sort by highest and get the suffixes and their counts
        ext_counts = extension_counts.most_common()
        if len(ext_counts) > 1:
            # Nicely format the list of files and their extensions
            for i, (ext, count) in enumerate(ext_counts):
                if i == (len(ext_counts) - 1):
                    txt += f"and {count} {ext}"
                else:
                    txt += f"{count} {ext}, "
        else:
            txt += f"{ext_counts[0][1]} {ext_counts[0][0]}"
        txt += f" file{'s' if sum(extension_counts.values()) > 1 else ''}."
        self.img_counts.setText(txt)
        # Reset the images loaded button text
        self.view_img_btn.setText(self.init_view_msg)

    def clear_directory(self):
        """
        Clears the selected directory and resets the image counts.
        """
        # Reset selected images and their paths
        self.image_path_dict = {}
        # Reset image count text
        self.img_counts.setText(self.init_file_msg)
        # Reset the images loaded button text
        self.view_img_btn.setText(self.init_view_msg)
        # Remove Image layers from napari viewer
        img_layers = [i for i in self.viewer.layers if isinstance(i, Image)]
        for layer in img_layers:
            self.viewer.layers.remove(layer)

    def specify_url(self):
        """
        Allow user to specify a URL to e.g. a Zarr file to use.

        Considerations here are around how to handle the data once loaded.
        May require napari-ome-zarr plugin.
        """
        raise NotImplementedError
