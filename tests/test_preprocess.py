"""Tests for the preprocessing subwidget.

The widget is driven in isolation (``parent=None``) with ``_apply_dim_state``
called directly, rather than through loaded images. The bugs these cover
produce plausible wrong *values* rather than errors, and several repair
themselves once the widget is touched, so they resist checking by hand.
"""

import pytest
from aiod_utils.preprocess import get_all_preprocess_methods

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
