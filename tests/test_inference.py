from typing import NamedTuple

import numpy as np
import pytest
from napari import run as napari_run
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
from skimage import io

# Smoke test: one representative (task, model display name, variant) combination.
SMOKE_PARAMS = [
    ("mito", "Empanada", "MitoNet v1"),
]


def _load_manifests():
    from aiod_registry import load_manifests

    return load_manifests(filter_access=True).values()


def _build_one_variant_per_model_params() -> list[tuple[str, str, str]]:
    """Build one (task, model_display_name, variant) entry per (task, model) pair.

    Takes the first available variant for each model under each task, giving
    broader coverage than the smoke test without exhausting every version.
    """
    params = []
    for manifest in _load_manifests():
        # Collect the first version name that appears for each task across all versions
        first_version_per_task: dict[str, str] = {}
        for version_name, version in manifest.versions.items():
            for task_name in version.tasks:
                if task_name not in first_version_per_task:
                    first_version_per_task[task_name] = version_name
        for task_name, version_name in first_version_per_task.items():
            params.append((task_name, manifest.name, version_name))
    return params


def _build_all_model_params() -> list[tuple[str, str, str]]:
    """Build the full (task, model_display_name, variant) list from installed manifests.

    Uses the same traversal as ModelWidget.extract_model_info so the display
    names and version strings always match what the UI dropdowns contain.
    """
    params = []
    for manifest in _load_manifests():
        for version_name, version in manifest.versions.items():
            for task_name in version.tasks:
                params.append((task_name, manifest.name, version_name))
    return params


def pytest_generate_tests(metafunc):
    """Parametrize task/model/variant at one of three levels of coverage:

    - (default)       SMOKE_PARAMS: one hardcoded combo as a quick sanity check.
    - --one-model:     a single user-specified (task, model, variant) combo.
    - --one-per-model: first variant for every (task, model) pair in the manifests.
    - --full-models:  every (task, model, variant) combination in the manifests.
    """
    if {"task", "model", "variant"}.issubset(metafunc.fixturenames):
        one_model = metafunc.config.getoption("--one-model")
        if one_model:
            parts = one_model.split(",", maxsplit=2)
            if len(parts) != 3:
                raise ValueError(
                    f"--one-model expects 'task,model,variant', got: {one_model!r}"
                )
            params = [(parts[0].strip(), parts[1].strip(), parts[2].strip())]
        elif metafunc.config.getoption("--full-models"):
            params = _build_all_model_params()
        elif metafunc.config.getoption("--one-per-model"):
            params = _build_one_variant_per_model_params()
        else:
            params = SMOKE_PARAMS
        metafunc.parametrize("task,model,variant", params)


# Fixtures
@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """Temporary directory for test image files."""
    d = tmp_path_factory.mktemp("aiod_cache") / "test_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def dummy_images(base_dir):
    """Create dummy TIFF images covering 2D/3D and single/multi-channel cases."""
    rng = np.random.default_rng(seed=0)
    images = {
        "dim2.tif": rng.integers(0, 256, (6, 6), dtype=np.uint8),
        "dim3.tif": rng.integers(0, 256, (6, 6, 6), dtype=np.uint8),
        "dim2_ch3.tif": rng.integers(0, 256, (6, 6, 3), dtype=np.uint8),
        "dim3_ch3.tif": rng.integers(0, 256, (6, 6, 6, 3), dtype=np.uint8),
    }
    paths = []
    for name, arr in images.items():
        p = base_dir / name
        io.imsave(p, arr)
        paths.append(p)
    yield paths
    for p in paths:
        if p.exists():
            p.unlink()


class InferenceFixture(NamedTuple):
    viewer: object
    widget: object


@pytest.fixture
def pipeline_timeout(request) -> int:
    """Return the pipeline timeout in seconds from the --pipeline-timeout CLI option."""
    return int(request.config.getoption("--pipeline-timeout"))


@pytest.fixture
def inference_widget(make_napari_viewer_proxy, base_dir, monkeypatch):
    """Load the AI on Demand Inference dock widget into the viewer.

    Redirects the Nextflow base directory to the shared temp dir so no data
    is written to the user's cache. The original base directory is restored
    after each test.

    Also suppresses store_settings so the user's settings file is never
    overwritten with test-run values.
    """
    viewer = make_napari_viewer_proxy()
    _, plugin_widget = viewer.window.add_plugin_dock_widget("aiod-napari", "Inference")
    monkeypatch.setattr(plugin_widget, "store_settings", lambda: None)
    return InferenceFixture(viewer=viewer, widget=plugin_widget)


class TestTaskModelWidget:
    """TaskWidget was merged into ModelWidget; these guard the merged behaviour."""

    def test_task_dropdown_updates_model_options(self, inference_widget):
        model_widget = inference_widget.widget.subwidgets["model"]

        from aiod_registry import TASK_NAMES

        # Nothing selected yet: placeholders in both dropdowns.
        assert model_widget.task_dropdown.itemText(0) == model_widget.task_name_init
        assert model_widget.model_dropdown.itemText(0) == model_widget.model_name_init

        task_name = next(iter(TASK_NAMES))
        task_index = model_widget.task_dropdown.findText(TASK_NAMES[task_name])
        assert task_index != -1

        model_widget.task_dropdown.setCurrentIndex(task_index)
        model_widget.on_task_select()

        assert inference_widget.widget.selected_task == task_name
        # Model dropdown should now list real models for this task, not the placeholder.
        assert model_widget.model_dropdown.itemText(0) != model_widget.model_name_init

    def test_task_placeholder_removed_after_selection(self, inference_widget):
        """Once a real task is chosen, the placeholder must not be reselectable.

        Regression test: previously the model family/version dropdowns retained
        stale entries if a user could get back to the "Select a task" placeholder.
        """
        model_widget = inference_widget.widget.subwidgets["model"]

        from aiod_registry import TASK_NAMES

        task_name = next(iter(TASK_NAMES))
        task_index = model_widget.task_dropdown.findText(TASK_NAMES[task_name])
        model_widget.task_dropdown.setCurrentIndex(task_index)
        model_widget.on_task_select()

        assert model_widget.task_dropdown.findText(model_widget.task_name_init) == -1

    def test_get_config_params_includes_task(self, inference_widget):
        model_widget = inference_widget.widget.subwidgets["model"]
        nxf_params = {
            "task": "mito",
            "model": "empanada_mito",
            "model_type": "v1",
            "model_config": "/fake/path.yaml",
        }

        assert model_widget.get_config_params(nxf_params) == {
            "task": "mito",
            "name": "empanada_mito",
            "model_type": "v1",
            "model_config": "/fake/path.yaml",
        }

    def test_load_config_file_migrates_legacy_top_level_task(
        self, inference_widget, monkeypatch
    ):
        """Configs saved before the TaskWidget merge stored task as a top-level key.

        load_config_file must fold it into the model config so ModelWidget (which
        now owns task selection) can still pick it up.
        """
        plugin_widget = inference_widget.widget
        model_widget = plugin_widget.subwidgets["model"]

        received_configs = []
        monkeypatch.setattr(
            model_widget, "load_config", lambda config: received_configs.append(config)
        )

        legacy_config = {
            "task": "mito",
            "model": {
                "name": "empanada_mito",
                "model_type": "v1",
                "model_config": "/fake/path.yaml",
            },
        }
        plugin_widget.load_config_file(legacy_config)

        assert received_configs == [
            {
                "name": "empanada_mito",
                "model_type": "v1",
                "model_config": "/fake/path.yaml",
                "task": "mito",
            }
        ]

    def test_load_config_file_new_format_task_untouched(
        self, inference_widget, monkeypatch
    ):
        """New-format configs already nest task under model; migration must be a no-op."""
        plugin_widget = inference_widget.widget
        model_widget = plugin_widget.subwidgets["model"]

        received_configs = []
        monkeypatch.setattr(
            model_widget, "load_config", lambda config: received_configs.append(config)
        )

        new_config = {
            "model": {
                "task": "nucleus",
                "name": "empanada_mito",
                "model_type": "v1",
                "model_config": "/fake/path.yaml",
            },
        }
        plugin_widget.load_config_file(new_config)

        assert received_configs == [new_config["model"]]


@pytest.mark.slow
class TestInferenceWorkflow:
    def test_full_inference_pass(
        self,
        inference_widget,
        dummy_images,
        pipeline_timeout,
        task,
        model,
        variant,
    ):
        """Run one full inference pipeline pass for the given task/model/variant."""
        napari_viewer, plugin_widget = (
            inference_widget.viewer,
            inference_widget.widget,
        )

        plugin_widget.subwidgets["data"].update_file_count(paths=dummy_images)
        plugin_widget.subwidgets["data"].view_images()

        # Select task
        from aiod_registry import TASK_NAMES

        task_dropdown = plugin_widget.subwidgets["model"].task_dropdown
        task_index = task_dropdown.findText(TASK_NAMES[task])
        assert task_index != -1, (
            f"Task '{task}' not found in dropdown options: "
            f"{[task_dropdown.itemText(i) for i in range(task_dropdown.count())]}"
        )
        task_dropdown.setCurrentIndex(task_index)
        plugin_widget.subwidgets["model"].on_task_select()

        # Select model
        model_dropdown = plugin_widget.subwidgets["model"].model_dropdown
        model_index = model_dropdown.findText(model)
        assert model_index != -1, (
            f"Model '{model}' not found in dropdown options: "
            f"{[model_dropdown.itemText(i) for i in range(model_dropdown.count())]}"
        )
        model_dropdown.setCurrentIndex(model_index)
        plugin_widget.subwidgets["model"].on_model_select()

        # Select variant
        variant_dropdown = plugin_widget.subwidgets["model"].model_version_dropdown
        variant_index = variant_dropdown.findText(variant)
        assert variant_index != -1, (
            f"Variant '{variant}' not found in dropdown options: "
            f"{[variant_dropdown.itemText(i) for i in range(variant_dropdown.count())]}"
        )
        variant_dropdown.setCurrentIndex(variant_index)
        plugin_widget.subwidgets["model"].on_model_version_select()

        overwrite_btn = plugin_widget.subwidgets["nxf"].overwrite_btn
        overwrite_btn.setChecked(True)
        assert overwrite_btn.isChecked()

        pipeline_done = False
        timed_out = False

        def _quit_app():
            app = QApplication.instance()
            if app is not None:
                app.quit()

        def on_pipeline_finished():
            nonlocal pipeline_done
            pipeline_done = True
            napari_viewer.close()
            _quit_app()

        def on_pipeline_failed():
            nonlocal pipeline_done
            pipeline_done = True
            napari_viewer.close()
            _quit_app()
            pytest.fail("Inference pipeline failed")

        def on_timeout():
            nonlocal pipeline_done, timed_out
            if not pipeline_done:
                pipeline_done = True
                timed_out = True
                napari_viewer.close()
                _quit_app()

        QTimer.singleShot(pipeline_timeout * 1000, on_timeout)

        plugin_widget.subwidgets["nxf"].pipeline_finished.connect(on_pipeline_finished)
        plugin_widget.subwidgets["nxf"].pipeline_failed.connect(on_pipeline_failed)

        def run_pipeline():
            plugin_widget.subwidgets["nxf"].nxf_run_btn.click()

        plugin_widget.subwidgets["data"].images_loaded.connect(run_pipeline)

        napari_run()

        if timed_out:
            pytest.fail(
                f"Inference pipeline timed out after {pipeline_timeout}s "
                f"(task={task!r}, model={model!r}, variant={variant!r})"
            )


class TestSequenceParams:
    """List/tuple params are typed as comma-separated text, so they must be
    split rather than cast wholesale.

    Regression test: `tuple("2,2")` yields `('2', ',', '2')` (a per-character
    split) and `yaml.dump` then writes a `!!python/tuple` tag that
    `yaml.safe_load` cannot read. Affects StarDist's `n_tiles` and PanSeg's
    `patch`, `patch_halo` and `ws_pixel_pitch`.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2,2", [2, 2]),
            ("2, 2", [2, 2]),
            (" 4,4,1 ", [4, 4, 1]),
            ("2", [2]),
            ("1.5,1.5", [1.5, 1.5]),
            # Round-trips a value rendered by str(), as fill_ui_from_config does
            ("[2, 2]", [2, 2]),
            ("(2, 2)", [2, 2]),
            # Empty/degenerate input means "unset", not an empty sequence
            ("", None),
            ("   ", None),
            (",", None),
            ("2,,2", [2, 2]),
        ],
    )
    def test_parse_sequence_param(self, text, expected):
        from aiod_napari.inference.model_selection import parse_sequence_param

        assert parse_sequence_param(text) == expected

    def test_never_casts_string_per_character(self):
        """The exact failure from the review: tuple("2,2") -> ('2', ',', '2')."""
        from aiod_napari.inference.model_selection import parse_sequence_param

        assert tuple("2,2") == ("2", ",", "2")  # the old behaviour
        assert parse_sequence_param("2,2") == [2, 2]

    def test_yaml_round_trip_is_safe_loadable(self):
        """Emitted sequences must survive yaml.safe_load with no python tags."""
        import yaml

        from aiod_napari.inference.model_selection import parse_sequence_param

        dumped = yaml.dump({"n_tiles": parse_sequence_param("2,2")})
        assert "!!python/tuple" not in dumped
        assert yaml.safe_load(dumped) == {"n_tiles": [2, 2]}

    def test_format_round_trips_through_parse(self):
        from aiod_napari.inference.model_selection import (
            format_sequence_param,
            parse_sequence_param,
        )

        assert format_sequence_param([2, 2]) == "2, 2"
        assert parse_sequence_param(format_sequence_param([4, 4, 1])) == [4, 4, 1]

    def test_n_tiles_from_widget_is_yaml_safe(self, inference_widget):
        """End-to-end: typing "2,2" into StarDist's n_tiles box must emit a
        plain list, not the per-character tuple ('2', ',', '2')."""
        import yaml
        from aiod_registry import TASK_NAMES

        model_widget = inference_widget.widget.subwidgets["model"]

        task_index = model_widget.task_dropdown.findText(TASK_NAMES["nuclei"])
        if task_index == -1:
            pytest.skip("nuclei task unavailable")
        model_widget.task_dropdown.setCurrentIndex(task_index)
        model_widget.on_task_select()

        model_index = model_widget.model_dropdown.findText("StarDist")
        if model_index == -1:
            pytest.skip("StarDist unavailable in this registry")
        model_widget.model_dropdown.setCurrentIndex(model_index)
        model_widget.on_model_select()
        model_widget.on_model_version_select()

        key = model_widget.get_task_model_variant(executed=False)
        n_tiles_widget = model_widget.model_param_dict[key]["Number of Tiles"]["value"]
        n_tiles_widget.setText("2,2")

        config = model_widget.create_config_params(task_model_version=key)
        assert config["n_tiles"] == [2, 2]

        dumped = yaml.dump(config)
        assert "!!python/tuple" not in dumped
        assert yaml.safe_load(dumped)["n_tiles"] == [2, 2]
