from collections import Counter
from pathlib import Path
from typing import Optional, Union

import napari
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
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
Select data to be used as input to the model.

Images can also be opened, or dragged into napari as normal. The selection will be updated accordingly. The 'Reset selection' button can be used to clear all images.
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
        counter = 0
        if self.viewer.layers:
            for img_layer in self.viewer.layers:
                if isinstance(img_layer, Image):
                    try:
                        img_path = Path(img_layer.source.path)
                        self.image_path_dict[img_path.stem] = img_path
                        counter += 1
                    except AttributeError:
                        show_info(
                            f"Cannot extract path for image layer {img_layer}. Please add manually using the buttons."
                        )
                        continue
        # If all pre-existing image layers have been added, set loaded flag
        if len(self.image_path_dict) == counter:
            self.existing_loaded = True
        else:
            self.existing_loaded = False
        # Create a button to select individual images from
        self.img_btn = QPushButton("Select image\nfiles")
        self.img_btn.clicked.connect(self.browse_imgs_files)
        self.img_btn.setToolTip(
            format_tooltip(
                "Select individual image files to use as input to the model."
            )
        )
        self.layout().addWidget(self.img_btn, 0, 0)
        # Create a button to navigate to a directory to take images from
        self.dir_btn = QPushButton("Select image\ndirectory")
        self.dir_btn.clicked.connect(self.browse_imgs_dir)
        self.dir_btn.setToolTip(
            format_tooltip(
                "Select folder/directory of images to use as input to the model."
            )
        )
        self.layout().addWidget(self.dir_btn, 0, 1)
        # Create a button to clear selected directory
        self.clear_dir_btn = QPushButton("Reset\nselection")
        self.clear_dir_btn.clicked.connect(self.clear_directory)
        self.clear_dir_btn.setToolTip(
            format_tooltip(
                "Reset selection of images (clears all images in the viewer)."
            )
        )
        self.layout().addWidget(self.clear_dir_btn, 0, 2)
        # Add an output to show the counts
        self.init_file_msg = "No files selected or added to Napari."
        self.img_counts = QLabel(self.init_file_msg)
        self.img_counts.setWordWrap(True)
        self.layout().addWidget(self.img_counts, 1, 0, 1, 3)

        # Run the file counter if there are images already loaded
        if len(self.image_path_dict) > 0:
            self.update_file_count()
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
            # Switch flag to signify the image has been loaded
            # Adding via drag+drop blocks the UI, so it's fine to do here and not when adding begins
            self.parent.subwidgets["nxf"].all_loaded = True

    def on_layer_removed(self, event):
        """
        Triggered whenever a layer is removed from the viewer.

        Checks if the layer is an image, and if so, removes it from the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = event.value.source.path
            # Remove from the list of images
            if img_path is not None:
                if Path(img_path).stem in self.image_path_dict:
                    del self.image_path_dict[Path(img_path).stem]
            else:
                if event.value.name in self.image_path_dict:
                    del self.image_path_dict[event.value.name]
            # Update file count with image removed
            self.update_file_count()

    def browse_imgs_files(self):
        """
        Opens a dialog for selecting images to segment.
        """
        # TODO: Implement a cache that stores the last directory used?
        # Should this cache persist across sessions, or is that invasive?
        fnames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more images",
            str(Path.home()),
            "",
        )
        if fnames != []:
            self.update_file_count(paths=fnames)
            self.view_images(imgs_to_load=fnames)

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
            all_paths = list(Path(result).glob("*"))
            self.update_file_count(paths=all_paths)
            self.view_images(imgs_to_load=all_paths)

    def view_images(
        self, imgs_to_load: Optional[list[Union[Path, str]]] = None
    ):
        """
        Loads the selected images into napari for viewing (in separate threads).
        """
        # Ensure Nextflow subwidget knows not all images are loaded until this func returns
        self.parent.subwidgets["nxf"].all_loaded = False
        # Return if there's nothing to show
        if len(self.image_path_dict) == 0:
            return
        if imgs_to_load is None:
            # Check if there are images to load that haven't been already
            viewer_imgs = [
                Path(i.name).stem
                for i in self.viewer.layers
                if isinstance(i, Image)
            ]
            imgs_to_load = [
                v
                for k, v in self.image_path_dict.items()
                if k not in viewer_imgs
            ]
        # If giving paths, double-check they aren't already loaded somehow
        elif imgs_to_load:
            remove_fnames = []
            for fname in imgs_to_load:
                if Path(fname).stem in self.viewer.layers:
                    remove_fnames.append(fname)
            imgs_to_load = [i for i in imgs_to_load if i not in remove_fnames]
        # Selecting no images will cause imgs_to_load=False, I think?
        else:
            return
        if len(imgs_to_load) == 0:
            return
        # Reset counter
        self.load_img_counter = 0

        # Create separate thread worker to avoid blocking
        @thread_worker(
            connect={
                "returned": self._add_image,
                "finished": self._finished_loading,
            }
        )
        def _load_image(fpath):
            return skimage.io.imread(fpath), Path(fpath)

        # Load each image in a separate thread
        for fpath in imgs_to_load:
            _load_image(fpath)
        # NOTE: This does not work well for a directory of large images on a remote directory
        # But that would trigger loading GBs into memory over network, which is...undesirable
        self.loading_txt = f" (loading {len(imgs_to_load)} image{'s' if len(imgs_to_load) > 1 else ''}...)"
        self.img_counts.setText(self.img_counts.text() + self.loading_txt)

    def _add_image(self, res):
        """
        Adds an image to the viewer when loaded, using its filepath as the name.
        """
        img, fpath = res
        # Add the image to the overall dict
        self.image_path_dict[fpath.stem] = fpath
        self.viewer.add_image(img, name=fpath.stem)

    def _finished_loading(self):
        """Signify to user that all images have been loaded."""
        self.img_counts.setText(
            self.img_counts.text().replace(
                self.loading_txt, " (all images loaded)."
            )
        )
        # Ensure Nextflow subwidget knows everything is loaded to extract metadata
        self.parent.subwidgets["nxf"].all_loaded = True
        # Also reset the viewer itself to ensure images are visible
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
        txt += f" file{'s' if sum(extension_counts.values()) > 1 else ''}"
        self.img_counts.setText(txt)

    def clear_directory(self):
        """
        Clears the selected directory and resets the image counts.
        """
        # Reset selected images and their paths
        self.image_path_dict = {}
        # Reset image count text
        self.img_counts.setText(self.init_file_msg)
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
