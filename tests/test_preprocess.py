"""Tests for the preprocessing subwidget.

The widget is driven in isolation (``parent=None``) with ``_apply_dim_state``
called directly, rather than through loaded images. The bugs these cover
produce plausible wrong *values* rather than errors, and several repair
themselves once the widget is touched, so they resist checking by hand.
"""

import numpy as np
import pytest
from aiod_utils.preprocess import get_all_preprocess_methods, run_preprocess

from aiod_napari.inference.preprocess import PreprocessWidget


@pytest.fixture
def preprocess_widget(make_napari_viewer_proxy):
    """A standalone PreprocessWidget with no parent plugin widget."""
    return PreprocessWidget(viewer=make_napari_viewer_proxy())


def _params(widget, method):
    return widget.preprocess_boxes[method]["params"]


def test_spinboxes_reject_values_the_backend_rejects(preprocess_widget):
    """Numeric params are bounded by their metadata, not by Qt's 0-99 default.

    Zero is invalid nearly everywhere: block_reduce and square(0) both reject it.
    """
    methods = get_all_preprocess_methods()

    size = _params(preprocess_widget, "Filter")["size"]
    assert size.minimum() == methods["Filter"]["params"]["size"]["min"] == 1
    assert size.maximum() >= 100

    # block_size is a list-backed param, so every subwidget takes the bounds
    for factor in _params(preprocess_widget, "Downsample")["block_size"]:
        assert factor.minimum() == 1
        assert factor.maximum() >= 1000

    for tile in _params(preprocess_widget, "CLAHE")["tileGridSize"]:
        assert tile.minimum() == 1


def test_spinboxes_accept_values_above_the_qt_default(preprocess_widget):
    """setValue past 99 must survive, not silently clamp.

    This is the path a config takes through _load_options_into_ui.
    """
    size = _params(preprocess_widget, "Filter")["size"]
    size.setValue(100)
    assert size.value() == 100

    clip = _params(preprocess_widget, "CLAHE")["clipLimit"]
    clip.setValue(40.0)
    assert clip.value() == pytest.approx(40.0)


def test_defaults_survive_being_bounded(preprocess_widget):
    """Every widget still holds its declared default after configuration."""
    for method, method_def in get_all_preprocess_methods().items():
        for param_name, param_def in method_def["params"].items():
            widget = _params(preprocess_widget, method)[param_name]
            default = param_def["default"]
            if isinstance(widget, list):
                assert [w.value() for w in widget] == list(default)
            elif hasattr(widget, "value"):
                assert widget.value() == pytest.approx(default)


def _tick(widget, method):
    """Tick a method's group box the way a user click would, callback included."""
    widget.preprocess_boxes[method]["box"].setChecked(True)
    widget.on_click_preprocess(method)()


def test_depth_factor_is_neutralised_for_2d_images(preprocess_widget):
    """A depth factor left over from a 3D image must not reach a 2D run.

    A disabled spin box keeps its value, and extract_options still reads it.
    """
    w = preprocess_widget
    w._apply_dim_state("3d")
    _tick(w, "Downsample")
    block = w.preprocess_boxes["Downsample"]["params"]["block_size"]
    block[0].setValue(4)

    w._apply_dim_state("2d")

    assert not block[0].isEnabled()
    extracted = w.extract_options()[0]["params"]["block_size"]
    assert extracted[0] == 1
    # The in-plane factors are dimension-agnostic and must survive untouched
    assert extracted[1:] == [2, 2]


def test_depth_factor_survives_unticking(preprocess_widget):
    """Only dimensionality clears the depth factor, never unticking the box.

    Both states disable the box, so keying off isEnabled would discard a value.
    """
    w = preprocess_widget
    w._apply_dim_state("3d")
    _tick(w, "Downsample")
    block = w.preprocess_boxes["Downsample"]["params"]["block_size"]
    block[0].setValue(4)

    w.preprocess_boxes["Downsample"]["box"].setChecked(False)
    w.on_click_preprocess("Downsample")()

    assert block[0].value() == 4


def test_mixed_dimensions_unticks_and_dequeues(preprocess_widget):
    """A method disabled for mixed dimensions must not still be queued to run.

    setEnabled(False) leaves a QGroupBox ticked; extract_options walks order_list.
    """
    w = preprocess_widget
    w._apply_dim_state("3d")
    _tick(w, "Filter")
    assert w.order_list == ["Filter"]

    w._apply_dim_state("mixed")

    box = w.preprocess_boxes["Filter"]["box"]
    assert not box.isEnabled()
    assert not box.isChecked()
    assert w.order_list is None
    assert w.preprocess_order.text() == w.init_order
    assert w.extract_options() is None


def test_mixed_dimensions_leaves_dimension_agnostic_methods_alone(preprocess_widget):
    """Only methods declaring requires_uniform_dims are withdrawn."""
    w = preprocess_widget
    _tick(w, "CLAHE")

    w._apply_dim_state("mixed")

    box = w.preprocess_boxes["CLAHE"]["box"]
    assert box.isEnabled()
    assert box.isChecked()
    assert w.order_list == ["CLAHE"]


def test_unticking_everything_restores_the_empty_sentinel(preprocess_widget):
    """order_list must return to None, not an empty list.

    get_all_options reads a non-None extract_options as unsaved work and warns.
    """
    w = preprocess_widget
    _tick(w, "Filter")

    w.preprocess_boxes["Filter"]["box"].setChecked(False)
    w.on_click_preprocess("Filter")()

    assert w.order_list is None
    assert w.preprocess_order.text() == w.init_order
    assert w.extract_options() is None


def _footprint(widget):
    return widget.preprocess_boxes["Filter"]["params"]["footprint"]


def test_reset_keeps_a_usable_footprint(preprocess_widget):
    """Resetting must not leave the combo on Qt's out-of-range -1."""
    w = preprocess_widget
    w._apply_dim_state("2d")

    w._reset_preprocess()

    combo = _footprint(w)
    assert combo.currentIndex() != -1
    assert combo.currentText() != ""


@pytest.mark.parametrize(("state", "expected"), [("2d", "disk"), ("3d", "ball")])
def test_footprint_falls_back_to_the_preferred_shape(
    preprocess_widget, state, expected
):
    """The fallback follows values_by_dim's ordering, not the unfiltered list's.

    The unfiltered list lands on square in 2D and cube in 3D, neither of which
    is the analogue of the disk default.
    """
    w = preprocess_widget
    w._apply_dim_state(state)
    w._reset_preprocess()

    assert _footprint(w).currentText() == expected


def test_config_with_an_unusable_footprint_warns_and_stays_runnable(
    preprocess_widget, monkeypatch
):
    """A 3D footprint in a config loaded against a 2D image must not go quiet.

    setChecked emits no clicked, so nothing re-runs the dimension logic to
    repair a blanked combo before the footprint reaches execution.
    """
    warned = []
    monkeypatch.setattr("aiod_napari.inference.preprocess.show_warning", warned.append)
    w = preprocess_widget
    w._apply_dim_state("2d")

    w.load_config(
        [
            [
                {
                    "name": "Filter",
                    "params": {"footprint": "ball", "size": 5, "method": "median"},
                }
            ]
        ]
    )

    combo = _footprint(w)
    assert combo.currentIndex() != -1
    assert combo.currentText() in ("disk", "square")
    assert any("ball" in str(m) for m in warned)
    # The substituted options have to actually run on the image in play
    run_preprocess(
        np.zeros((32, 32), dtype=np.uint8), w.extract_options(), only_check=True
    )


def test_options_round_trip_through_the_ui(preprocess_widget):
    """What _load_options_into_ui accepts, extract_options must give back."""
    w = preprocess_widget
    w._apply_dim_state("3d")
    options = [
        {"name": "Downsample", "params": {"block_size": [2, 4, 4], "method": "mean"}},
        {
            "name": "Filter",
            "params": {"footprint": "ball", "size": 7, "method": "median"},
        },
    ]

    w._load_options_into_ui(options)

    assert w.extract_options() == options
