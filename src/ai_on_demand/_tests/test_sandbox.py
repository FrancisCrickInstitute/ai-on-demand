import pytest
import torch
from napari import Viewer, run
from qtpy.QtCore import QTimer
from skimage import io
import numpy as np
from pathlib import Path
import os


dummy_images = ["dim2", "dim3", "dim2_ch3", "dim3_ch3"]


def prep_dummy_images(base_dir):
    save_path = base_dir / "aiod_cache" / "test_cache"
    save_path.mkdir(parents=True, exist_ok=True)

    image_dim2 = torch.randint(0, 256, (6, 6), dtype=torch.uint8).numpy()
    image_dim3 = torch.randint(0, 256, (6, 6, 6), dtype=torch.uint8).numpy()
    image_dim2_ch3 = torch.randint(
        0, 256, (6, 6, 3), dtype=torch.uint8
    ).numpy()
    image_dim3_ch3 = torch.randint(
        0, 256, (6, 6, 6, 3), dtype=torch.uint8
    ).numpy()

    io.imsave(save_path / "dim2.tif", image_dim2)
    io.imsave(save_path / "dim3.tif", image_dim3)
    io.imsave(save_path / "dim2_ch3.tif", image_dim2_ch3)
    io.imsave(save_path / "dim3_ch3.tif", image_dim3_ch3)


def clean_up(base_dir):
    save_path = base_dir / "aiod_cache" / "test_cache"

    for image_name in dummy_images:
        fpath = save_path / (image_name + ".tif")
        if os.path.isfile(fpath):
            print("deleting file: ", fpath)
            os.remove(fpath)
    return None


def test_inference_workflow(task, model, variant):
    napari_viewer = Viewer()

    dock_widget, plugin_widget = napari_viewer.window.add_plugin_dock_widget(
        "ai-on-demand", "Inference"
    )

    max_time_duration = 300000  # given in ms = 5mins

    base_dir = Path(plugin_widget.subwidgets["nxf"].nxf_base_dir)
    print("this is the base dir:", base_dir)

    prep_dummy_images(base_dir)

    plugin_widget.subwidgets["data"].update_file_count(
        paths=[
            base_dir / "aiod_cache/test_cache" / "dim2.tif",
            base_dir / "aiod_cache/test_cache" / "dim3.tif",
            base_dir / "aiod_cache/test_cache" / "dim2_ch3.tif",
            base_dir / "aiod_cache/test_cache" / "dim3_ch3.tif",
        ]
    )

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

    # overwrite
    plugin_widget.subwidgets["nxf"].overwrite_btn.setChecked(True)

    # Assert that the widget is in a ready state (example: check overwrite is set)
    assert plugin_widget.subwidgets["nxf"].overwrite_btn.isChecked()

    def on_pipeline_finished(model=model, variant=variant):
        print("pipeline finished!")

        # To test for if the layers have been create we need to get the hash somehow?
        # as each hash is diff for each model and model variant

        # # prep params to be in layer text format
        # layer_variant = variant.replace(" ", "-")
        # layer_model = model.lower()

        # # these should exist
        # print("layer variant:", layer_variant)
        # for image_name in dummy_images:
        #     output_mask = napari_viewer.layers[
        #         f"{image_name}_masks_{task}-{layer_model}-{layer_variant}-43e45ccf"
        #     ].data

        napari_viewer.close()

    def on_pipeline_failed():
        napari_viewer.close()
        pytest.fail("Inference pipeline failed")

    plugin_widget.subwidgets["nxf"].pipeline_finished.connect(
        on_pipeline_finished
    )
    plugin_widget.subwidgets["nxf"].pipeline_failed.connect(on_pipeline_failed)

    def run_pipeline():
        plugin_widget.subwidgets["nxf"].nxf_run_btn.click()
        print("running the pipeline!")

    def timeout():
        # cancel pipeline & close viewer
        try:
            plugin_widget.subwidgets["nxf"].cancel_btn.click()
        except:
            print("No pipeline running!")
        napari_viewer.close()
        pytest.fail(f"Timeout duration:{max_time_duration}")

    QTimer.singleShot(
        3000, run_pipeline
    )  # possible issue if the image didn't load in 3 seconds? better to have this on an emit?
    QTimer.singleShot(max_time_duration, timeout)  # time out after 1 minute
    run()


base_dir = Path("/Users/ahmedn/.nextflow/aiod")
models = [
    "mito",
    "er",
    "ne",
    "everything",
    "nuclei",
    "cyto",
    "drop",
]  # need to remove the ones which are not impelmented

model_info = {
    # "mito": {"Empanada": ["MitoNet v1", "MitoNet Mini v1"]},
    # "everything": {"Segment Anything": ["default", "vit_h", "vit_l", "vit_b"]},
    "nuclei": {"Empanada": ["NucleoNet v1"]},
}

models = ["mito", "everything"]
if __name__ == "__main__":
    for task, model_params in model_info.items():
        for model, variants in model_params.items():
            for variant in variants:
                print(f"testing: {task},{model}, {variant}")
                test_inference_workflow(task, model, variant)
