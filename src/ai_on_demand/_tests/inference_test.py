import pytest
import torch
from napari import Viewer, run
from qtpy.QtCore import QTimer
from skimage import io
import numpy as np
from pathlib import Path


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


@pytest.fixture
def prep_dummy_images(base_dir):
    images = []
    image_dim2 = torch.randn(8, 8)
    image_dim3 = torch.randn(8, 8, 8)
    image_dim2_ch3 = torch.randn(8, 8, 3)
    image_dim3_ch3 = torch.randn(8, 8, 8, 3)

    # save all images to cache

    save_path = Path(base_dir) / "test_cache"

    io.imsave(image_dim2, save_path)
    io.imsave(image_dim3, save_path)
    io.imsave(image_dim2_ch3, save_path)
    io.imsave(image_dim3_ch3, save_path)
    return images


def clean_up():
    # remove the images
    return None


def test_inference_workflow(napari_viewer, inference_widget):
    max_time_duration = 60000  # msec = 60seconds
    plugin_widget = inference_widget

    base_dir = plugin_widget.subwidgets["nxf"].nxf_base_dir
    prep_dummy_images(base_dir)
    print("this is the base dir:", base_dir)
    breakpoint()

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
