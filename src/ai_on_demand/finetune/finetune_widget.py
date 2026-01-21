from typing import Optional, Union

import napari
from napari.qt.threading import thread_worker

from ai_on_demand.finetune import FinetuneParameters

from ai_on_demand.inference import (
    TaskWidget,
    DataWidget,
    ModelWidget,
    NxfWidget,
    ConfigWidget,
)

from ai_on_demand.widget_classes import MainWidget, SubWidget, QGroupBox
from ai_on_demand.utils import calc_param_hash


class Finetune(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer,
            title="Finetuning",
            tooltip="""
            Finetune existing models
                         """,
        )
        self.selected_task = None
        self.selected_model = None
        self.selected_variant = None
        self.executed_task = None
        self.executed_model = None
        self.executed_variant = None
        self.run_hash = None

        # Create radio buttons for selecting task (i.e. organelle)
        self.register_widget(
            TaskWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.register_widget(
            ModelWidget(
                viewer=self.viewer,
                variant="finetune",
                parent=self,
                expanded=False,
            )
        )

        # # Create the box for selecting the directory, showing img count etc.
        # self.register_widget(
        #     DataWidget(
        #         viewer=self.viewer,
        #         parent=self,
        #         expanded=False,
        #         variant="finetune",
        #     )
        # )
        # TODO: reevaluate if you can use the data widget for both image dir and masks nicely

        self.register_widget(
            FinetuneParameters(viewer=self.viewer, parent=self)
        )

        # Add the button for running the Nextflow pipeline
        self.register_widget(
            NxfWidget(
                viewer=self.viewer,
                parent=self,
                variant="finetune",
                expanded=False,
            )
        )

        self.subwidgets["nxf"].finetuned_model_ready.connect(
            self.subwidgets["finetune_params"].enable_add_model
        )

    def get_run_hash(self, nxf_params: dict):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        hashed_params = {}
        # Add model details
        hashed_params["task"] = nxf_params["task"]
        hashed_params["model"] = nxf_params["model"]
        hashed_params["variant"] = nxf_params["model_type"]
        # Add the model dictionary (hashed)
        hashed_params["model_hash"] = self.subwidgets["model"].model_param_hash
        # and Nextflow parameters that affect the output
        self.run_hash = calc_param_hash(hashed_params)
