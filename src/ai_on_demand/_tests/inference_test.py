import pytest
import torch
from napari import Viewer, run
from qtpy.QtCore import QTimer
from skimage import io
import numpy as np
from pathlib import Path
import os


# Fixtures


# Sometimes napari remains running even after The viewer is closed
@pytest.fixture(scope="session", autouse=True)
def ensure_qt_cleanup():
    yield
    try:
        run.quit()
    except Exception:
        pass


@pytest.fixture
def napari_viewer(request):
    viewer = Viewer()
    yield viewer
    try:
        viewer.close()
    except Exception:
        pass
    try:
        run.quit()
    except Exception:
        pass


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    # Use a temporary directory for test cache
    d = tmp_path_factory.mktemp("aiod_cache") / "test_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def dummy_images(base_dir):
    image_dim2 = torch.randint(0, 256, (6, 6), dtype=torch.uint8).numpy()
    image_dim3 = torch.randint(0, 256, (6, 6, 6), dtype=torch.uint8).numpy()
    image_dim2_ch3 = torch.randint(
        0, 256, (6, 6, 3), dtype=torch.uint8
    ).numpy()
    image_dim3_ch3 = torch.randint(
        0, 256, (6, 6, 6, 3), dtype=torch.uint8
    ).numpy()

    paths = [
        base_dir / "dim2.tif",
        base_dir / "dim3.tif",
        base_dir / "dim2_ch3.tif",
        base_dir / "dim3_ch3.tif",
    ]
    io.imsave(paths[0], image_dim2)
    io.imsave(paths[1], image_dim3)
    io.imsave(paths[2], image_dim2_ch3)
    io.imsave(paths[3], image_dim3_ch3)
    yield paths
    # remove all the params after the test
    for p in paths:
        if p.exists():
            p.unlink()


@pytest.fixture
def inference_widget(napari_viewer):
    dock_widget, plugin_widget = napari_viewer.window.add_plugin_dock_widget(
        "ai-on-demand", "Inference"
    )
    return plugin_widget


# store each task, model and variant - comment out to test specific tasks, models and variants


# QUICK mito + MitoNet test
model_info = {
    "mito": {"Empanada": ["MitoNet v1"]},
}


# TEST ALL MODELS
# model_info = {
# "mito": {"Empanada": ["MitoNet v1", "MitoNet Mini v1"]},
# "nuclei": {
#     "Cellpose": ["nuclei"],
#     "Cellpose-SAM": ["cpsam"],
#     "Empanada": ["NucleoNet v1"],
# },
# "everything": {
#     "Segment Anything": [
#         "default",
#         "vit_h",
#         "vit_l",
#         "vit_b",
#         "MedSAM",
#         "MicroSAM-Boundaries",
#         "MicroSAM-Organelles",
#     ],
#     "Segment Anything 2": [
#         "hiera_base",
#         "hiera_small",
#         "hiera_large",
#         "hiera_tiny",
#     ],
# "cyto": {"Cellpose": ["cyto3", "cyto1", "cyto3"]},
# "drop": {"Empanada": ["DropNet v1"]},
# },  # SAM models Usually take long to run
# }


# test for each task, model and variant
def pytest_generate_tests(metafunc):
    if {"task", "model", "variant"}.issubset(metafunc.fixturenames):
        argvalues = []
        for task, model_params in model_info.items():
            for model, variants in model_params.items():
                for variant in variants:
                    argvalues.append((task, model, variant))
        metafunc.parametrize("task,model,variant", argvalues)


# one full inference pass
def test_inference_workflow(
    napari_viewer, inference_widget, dummy_images, task, model, variant
):
    plugin_widget = inference_widget

    # Update file count with dummy images
    plugin_widget.subwidgets["data"].update_file_count(paths=dummy_images)
    plugin_widget.subwidgets["data"].view_images()

    # Select task
    plugin_widget.subwidgets["task"].task_buttons[task].click()

    # Select model in dropdown (QComboBox)
    model_dropdown = plugin_widget.subwidgets["model"].model_dropdown
    model_index = model_dropdown.findText(model)
    assert (
        model_index != -1
    ), f"Model '{model}' not found in dropdown options: {[model_dropdown.itemText(i) for i in range(model_dropdown.count())]}"
    model_dropdown.setCurrentIndex(model_index)
    plugin_widget.subwidgets["model"].on_model_select()

    # Select variant/version in dropdown (QComboBox)
    variant_dropdown = plugin_widget.subwidgets["model"].model_version_dropdown
    variant_index = variant_dropdown.findText(variant)
    assert (
        variant_index != -1
    ), f"Variant '{variant}' not found in dropdown options: {[variant_dropdown.itemText(i) for i in range(variant_dropdown.count())]}"
    variant_dropdown.setCurrentIndex(variant_index)
    plugin_widget.subwidgets["model"].on_model_version_select()

    # Overwrite
    plugin_widget.subwidgets["nxf"].overwrite_btn.setChecked(True)
    assert plugin_widget.subwidgets["nxf"].overwrite_btn.isChecked()

    finished = {"done": False}

    def on_pipeline_finished(model=model, variant=variant):
        finished["done"] = True
        print("pipeline finished!")
        # Optionally, check output layers here
        napari_viewer.close()
        run.quit()

    def on_pipeline_failed():
        finished["done"] = True
        napari_viewer.close()
        run.quit()
        pytest.fail("Inference pipeline failed")

    plugin_widget.subwidgets["nxf"].pipeline_finished.connect(
        on_pipeline_finished
    )
    plugin_widget.subwidgets["nxf"].pipeline_failed.connect(on_pipeline_failed)

    def run_pipeline():
        plugin_widget.subwidgets["nxf"].nxf_run_btn.click()
        print("running the pipeline!")

    plugin_widget.subwidgets["data"].images_loaded.connect(run_pipeline)

    def timeout():
        if not finished["done"]:
            try:
                plugin_widget.subwidgets["nxf"].cancel_btn.click()
            except Exception:
                pass
            napari_viewer.close()
            run.quit()
            pytest.fail("Timeout during inference workflow")

    QTimer.singleShot(600000, timeout)  # 10min timeout
    run()
