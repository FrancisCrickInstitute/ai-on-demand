"""
Basic widget tests for the ai-on-demand plugin.

These are the original simple tests, updated with proper mocking.
"""
import pytest
from unittest.mock import patch
from ai_on_demand.inference.inference_widget import Inference
from ai_on_demand.evaluation import Evaluation


def test_inference(make_napari_viewer, mock_plugin_manager, mock_aiod_utils):
    """Test basic inference widget creation."""
    viewer = make_napari_viewer()

    inf_widget = Inference(viewer)

    assert inf_widget is not None
    assert inf_widget.viewer == viewer


def test_evaluation(make_napari_viewer, mock_plugin_manager, mock_aiod_utils):
    """Test basic evaluation widget creation."""
    viewer = make_napari_viewer()

    eval_widget = Evaluation(viewer)

    assert eval_widget is not None
    assert eval_widget.viewer == viewer
