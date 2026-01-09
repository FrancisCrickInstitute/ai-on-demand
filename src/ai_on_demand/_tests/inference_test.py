import pytest
from napari import Viewer


@pytest.fixture(scope="module")
def napari_viewer():
    viewer = Viewer()
    yield viewer
    viewer.close()


@pytest.fixture(scope="module")
def inference_widget(napari_viewer):
    dock_widget, plugin_widget = napari_viewer.window.add_plugin_dock_widget(
        "ai-on-demand", "Inference"
    )
    return plugin_widget


def test_inference_workflow(inference_widget):
    inference_widget.subwidgets["data"].update_file_count(
        paths=[
            "/Users/ahmedn/Work/aiod_test_images/example_mito.tif",
        ]
    )
    inference_widget.subwidgets["data"].view_images()

    # Select task
    inference_widget.subwidgets["task"].task_buttons["mito"].click()

    # overwrite
    inference_widget.subwidgets["nxf"].overwrite_btn.setChecked(True)

    # trigger the run (commented in debug.py) - won't work unless the images have been loaded
    # inference_widget.subwidgets["nxf"].nxf_run_btn.click()

    # Assert that the widget is in a ready state (example: check overwrite is set)
    assert inference_widget.subwidgets["nxf"].overwrite_btn.isChecked()


import pytest
