import pytest
from napari import Viewer, run
from qtpy.QtCore import QTimer
from skimage import io
import numpy as np


@pytest.fixture(scope="module")
def napari_viewer():
    viewer = Viewer()
    yield viewer
    viewer.close()


@pytest.fixture
def inference_widget(napari_viewer):
    dock_widget, plugin_widget = napari_viewer.window.add_plugin_dock_widget(
        "ai-on-demand", "Inference"
    )
    return plugin_widget


def test_inference_workflow(napari_viewer, inference_widget):
    max_time_duration = 60000  # msec = 60seconds
    plugin_widget = inference_widget

    plugin_widget.subwidgets["data"].update_file_count(
        paths=[
            "-",
        ]
    )

    plugin_widget.subwidgets["data"].view_images()

    # Select task
    plugin_widget.subwidgets["task"].task_buttons["mito"].click()

    # overwrite
    plugin_widget.subwidgets["nxf"].overwrite_btn.setChecked(True)

    # Assert that the widget is in a ready state (example: check overwrite is set)
    assert plugin_widget.subwidgets["nxf"].overwrite_btn.isChecked()

    def on_pipeline_finished():
        print("pipeline finished!")
        output_mask = napari_viewer.layers[
            "2D_mito_masks_mito-empanada-MitoNet-v1-43e45ccf"
        ].data

        reference_mask = io.imread(
            "./test_images/masks/2D_mitonet_V1_masks.tif"
        )

        assert np.array_equal(reference_mask, output_mask)
        napari_viewer.close()
        run.quit()

    def on_pipeline_failed():
        pytest.fail("Inference pipeline failed")
        napari_viewer.close()
        run.quit()

    plugin_widget.subwidgets["nxf"].pipeline_finished.connect(
        on_pipeline_finished
    )
    plugin_widget.subwidgets["nxf"].pipeline_failed.connect(on_pipeline_failed)

    def run_pipeline():
        plugin_widget.subwidgets["nxf"].nxf_run_btn.click()
        print("running the pipeline!")

    def timeout():
        pytest.fail(f"Timeout duration:{max_time_duration}")

    QTimer.singleShot(
        3000, run_pipeline
    )  # possible issue if the image didn't load in 3 seconds? better to have this on an emit?
    QTimer.singleShot(max_time_duration, timeout)  # time out after 1 minute
    run()
