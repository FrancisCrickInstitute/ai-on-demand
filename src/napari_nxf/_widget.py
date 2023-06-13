from collections import Counter
from pathlib import Path
import time
from typing import Union

from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QFileDialog, QLabel, QLineEdit, QRadioButton, QGroupBox
from qtpy.QtGui import QPixmap
import qtpy.QtCore
import numpy as np
import skimage.io

import napari
from napari.qt.threading import thread_worker

# Available models, with displayable names
MODELS = {
    "unet": "UNet",
    "sam": "Segment Anything"
}
# Specify which models are available for each task
MODEL_DICT = {
    "mito": [k for k in MODELS if k != "sam"],
    "er": [k for k in MODELS if k != "sam"],
    "ne": [k for k in MODELS if k != "sam"],
    "sam": ["sam"],
}

class AIOnDemand(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Handy attributes to check things
        self.selected_task = None
        self.selected_model = None

        # Set the path to watch for masks
        self.mask_path = Path(__file__).parent / "nextflow" / "resources/usr/bin" / "sam_masks"

        # Set overall layout for widget
        self.setLayout(QVBoxLayout())

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        # Create radio buttons for selecting task (i.e. organelle)
        self.create_organelle_box()

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.create_model_box()

        # Create the box for selecting the directory, showing img count etc.
        self.create_dir_box()

        # Add the button for running the Nextflow pipeline
        self.create_nxf_button()

    def create_organelle_box(self):
        # Define the box and layout
        self.task_group = QGroupBox("Select organelle to segment:")
        self.task_layout = QVBoxLayout()
        # Define and set the buttons for the different tasks
        # With callbacks to change other options accoridngly
        self.task_buttons = {}
        for name, label in zip(
            ["mito", "er", "ne", "sam"],
            ["Mitochondria", "Endoplasmic Reticulum", "Nuclear Envelope", "Everything!"]
        ):
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            btn.clicked.connect(self.update_model_box)
            self.task_layout.addWidget(btn)
            self.task_buttons[name] = btn
        # Add the buttons under the overall group box
        self.task_group.setLayout(self.task_layout)
        # Add to main widget
        self.layout().addWidget(self.task_group)

    def create_model_box(self):
        self.model_group = QGroupBox("Select model to run:")
        self.model_layout = QVBoxLayout()
        # Define and set the buttons for each model
        self.model_sam = QRadioButton("Segment Anything")
        self.model_buttons = {}
        for name, label in MODELS.items():
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            # Necessary to allow for unchecking
            btn.setAutoExclusive(False)
            self.model_layout.addWidget(btn)
            self.model_buttons[name] = btn
        self.model_group.setLayout(self.model_layout)
        self.layout().addWidget(self.model_group)

    def update_model_box(self):
        '''The model box updates according to what's defined for each task
        '''
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
                self.selected_task = task_name
                # Get the models available for this task
                avail_models = MODEL_DICT[task_name]
                # Disable selection of all models not selected
                for model_name, model_btn in self.model_buttons.items():
                    if model_name not in avail_models:
                        # Grey out and disable ineligible options
                        model_btn.setEnabled(False)
                        model_btn.setStyleSheet("color: gray")
                        # Uncheck if already checked and no longer available
                        model_btn.setChecked(False)
                    else:
                        model_btn.setEnabled(True)
                        model_btn.setStyleSheet("")

    def _check_models(self):
        for model_name, model_btn in self.model_buttons.items():
            if model_btn.isChecked():
                self.selected_model = model_name
                return

    def create_dir_box(self):
        # TODO: Simultaneously allow for drag+dropping, probably a common use pattern
        self.dir_group = QGroupBox("Data selection:")
        self.dir_layout = QVBoxLayout()
        # Add an output to show the selected path
        self.images_dir_label = QLabel("Image folder: not selected.")
        self.images_dir_label.setWordWrap(True)
        # Create empty container for selected image filepaths
        self.all_img_files = None
        # Create empty container for the image directory
        self.images_dir = None
        # Create empty counter to show image load progress
        self.load_img_counter = 0
        # Create a button to navigate to a directory to take images from
        self.dir_btn = QPushButton("Select directory")
        self.dir_btn.clicked.connect(self.browse_directory)
        self.dir_btn.setToolTip("Select folder/directory of images to use as input to the model.")
        self.dir_layout.addWidget(self.dir_btn)
        # Add an output to show the counts
        self.img_counts = QLabel()
        self.img_counts.setWordWrap(True)
        self.img_counts.setText("No files selected.")
        self.dir_layout.addWidget(self.img_counts)
        # self.images_dir_label = QLineEdit("")  # TODO: Make it editable fo ruser input too
        self.dir_layout.addWidget(self.images_dir_label)
        # Add a button for viewing the images within napari
        # Optional as potentially taxing, and not necessary
        self.view_img_btn = QPushButton("View selected images")
        self.view_img_btn.setToolTip("Load selected images into napari to view.")
        self.view_img_btn.clicked.connect(self.view_images)
        self.dir_layout.addWidget(self.view_img_btn)
        # Sort out layout and add to main widget
        self.dir_group.setLayout(self.dir_layout)
        self.layout().addWidget(self.dir_group)

    def abspath(self, root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    def browse_directory(self):
        result = QFileDialog.getExistingDirectory(
            self,
            caption="Select image directory",
            directory=None
        )
        # If a new directory is selected, reset the load button text
        if result != self.images_dir:
            self.view_img_btn.setText("View selected images")
            self.view_img_btn.setEnabled(True)
        if result != "":
            self.images_dir = result
            self.images_dir_label.setText(f"Image folder:\n{self.images_dir}")
            self._count_files()

    def type_directory(self):
        '''Allow for user to type a directory?

        Will require checking that the path is valid, and using the ToolTip if it's not
        (or a similar pop-up/warning object) to raise the issue.
        '''
        return NotImplementedError

    def _count_files(self):
        '''Function to extract all the files in a given path,
        and return a count (broken down by extension)
        '''
        txt = ""
        # Get all the files in the given path
        self.all_img_files = list(Path(self.images_dir).glob("*"))
        # Get all the extensions in the path
        extension_counts = Counter([i.suffix for i in self.all_img_files])
        # Sort by highest and get the suffixes and their counts
        ext_counts = extension_counts.most_common()
        if len(ext_counts) > 1:
            # Nicely format the list of files and their extensions
            for i, (ext, count) in enumerate(ext_counts):
                if i == (len(ext_counts)-1):
                    txt += f"and {count} {ext}"
                else:
                    txt += f"{count} {ext}, "
        else:
            txt += f"{ext_counts[0][1]} {ext_counts[0][0]}"
        txt += " files selected."
        self.img_counts.setText(txt)

    def view_images(self):
        self.view_img_btn.setEnabled(False)
        # Return if there's nothing to show
        if self.all_img_files is None:
            return
        # Reset counter
        self.load_img_counter = 0
        # Create separate thread worker to avoid blocking
        @thread_worker(connect={"returned": self._add_image, "finished": self._reset_view_btn})
        def _load_image(fpath):
            return skimage.io.imread(fpath), fpath
        # Load each image in a separate thread
        for fpath in self.all_img_files:
            _load_image(fpath)
        # NOTE: This does not work well for a directory of large images on a remote directory
        # But that would trigger loading GBs into memory over network, which is...undesirable
        self.view_img_btn.setText("Loading...")

    def _add_image(self, res):
        img, fpath = res
        self.viewer.add_image(img, name=fpath.name)
        self.load_img_counter += 1
        self.view_img_btn.setText(f"Loading...({self.load_img_counter}/{len(self.all_img_files)} images loaded).")
        if self.load_img_counter == len(self.all_img_files):
            self.view_img_btn.setText(f"All ({self.load_img_counter}/{len(self.all_img_files)}) images loaded.")

    def _reset_view_btn(self):
        self.view_img_btn.setEnabled(True)

    def watch_mask_files(self):
        # Wait for at least one image to load as layers if not present
        if not self.viewer.layers:
            time.sleep(0.5)
        # Create the Labels layers for each image
        for fpath in self.all_img_files:
            # If images still not loaded, add dummy array
            try:
                img_shape = self.viewer.layers[f"{fpath.name}"].data.shape
            except KeyError:
                img_shape = (500,500)
            # Set the name following convention
            name = f"{fpath.stem}_masks"
            # Add a Labels layer for this file
            self.viewer.add_labels(
                np.zeros(img_shape, dtype=int),
                name=name,
                visible=False
            )
        # NOTE: Wrapper as self/class not available at runtime
        @thread_worker(connect={"yielded": self.update_masks})
        def _watch_mask_files(self):
            # Enable the watcher
            self.watcher_enabled = True
            # Initialize empty container for storing mask filepaths
            self.mask_fpaths = []
            # Loop and yield any changes infinitely while enabled
            while self.watcher_enabled:
                # Get all files
                current_files = list(self.mask_path.glob("*.npy"))
                if set(self.mask_fpaths) != set(current_files):
                    # Get the new files only
                    new_files = [i for i in current_files if i not in self.mask_fpaths]
                    # Update file list and yield the difference
                    self.mask_fpaths = current_files
                    if new_files:
                        yield new_files
                # Sleep until next check
                time.sleep(2)
                # Check all masks contain data for all slices
                masks_finished = [Path(i).stem[-3:] == "all" for i in current_files]
                # Get how many mask files there should be
                num_images = len(self.all_img_files)
                # If all images have complete masks, deactivate watcher
                print(masks_finished)
                print(len(masks_finished), num_images)
                if all(masks_finished) and (len(masks_finished) == num_images):
                    print("Deactivating watcher...")
                    self.watcher_enabled = False
        # Call the nested function
        _watch_mask_files(self)

    def update_masks(self, new_files):
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            mask_arr = np.load(f)
            # Extract the relevant Labels layer
            layer_name = f.stem.split("_masks")[0] + "_masks"
            label_layer = self.viewer.layers[layer_name]
            # label_layer.num_colours = mask_arr.max()+1
            # Insert mask data
            label_layer.data = mask_arr
            label_layer.visible = True

    def create_nxf_button(self):
        self.nxf_layout = QHBoxLayout()
        # Create a button to navigate to a directory to take images from
        self.nxf_btn = QPushButton("Run Pipeline!")
        self.nxf_btn.clicked.connect(self.run_pipeline)
        self.nxf_btn.setToolTip("Run the pipeline with the chosen organelle(s), model, and images.")
        self.nxf_layout.addWidget(self.nxf_btn)
        self.layout().addLayout(self.nxf_layout)

    def run_pipeline(self):
        # Start with some error checking to ensure that everything has been properly specified
        # Check a task/organelle has been selected
        if self.selected_task is None:
            raise ValueError("No task/organelle has been selected!")
        # Check a model has been selected
        self._check_models()
        if self.selected_model is None:
            raise ValueError("No model has been selected!")
        # Reset LayerList
        self.viewer.layers.clear()
        # Check a directory of images has been given
        # NOTE: They do not have to have been loaded, but to show feedback they will be loaded
        self.view_images()
        # TODO: Actually run the pipeline...

        # Start the mask file watcher
        self.watch_mask_files()

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
