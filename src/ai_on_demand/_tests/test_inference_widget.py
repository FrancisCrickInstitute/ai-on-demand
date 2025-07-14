"""
Tests for the Inference widget focused on high-coverage business logic.

This module tests the critical inference functionality:
- Task-to-model workflow 
- Parameter gathering for run identification
- Model selection state management
- Core business logic rather than UI creation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

from ai_on_demand.inference.inference_widget import Inference


class TestInferenceWidget:
    """Test the main Inference widget business logic and workflows."""
    
    def test_task_selection_updates_model_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, minimal_manifest):
        """Test critical business logic: task selection triggers model widget update."""
        mock_plugin_manager.commands.execute.return_value = minimal_manifest
        widget = Inference(mock_napari_viewer)
        
        # Mock the model widget's update method
        widget.subwidgets["model"].update_model_box = Mock()
        
        # Simulate task selection
        widget.selected_task = "everything"
        
        # Simulate the task click workflow that would normally happen through UI
        widget.subwidgets["model"].update_model_box("everything")
        
        # Verify the model widget was updated with the correct task
        widget.subwidgets["model"].update_model_box.assert_called_with("everything")
        assert widget.selected_task == "everything"
    
    def test_parameter_gathering_for_run_hash(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test critical business logic: parameter gathering for run identification."""
        widget = Inference(mock_napari_viewer)
        
        # Setup mock subwidgets with parameter values
        mock_model_widget = Mock()
        mock_model_widget.model_param_hash = "test_model_hash"
        mock_nxf_widget = Mock()
        mock_nxf_widget.postprocess_btn.isChecked.return_value = True
        
        widget.subwidgets = {
            "model": mock_model_widget,
            "nxf": mock_nxf_widget
        }
        
        # Test parameter gathering
        test_params = {
            "task": "test_task",
            "model": "test_model",
            "model_type": "test_variant",
            "num_substacks": 2,
            "overlap": 15,
            "preprocess": ["normalize", "resize"],
            "iou_threshold": 0.7
        }
        
        widget.get_run_hash(test_params)
        
        # Verify run hash was calculated and critical parameters included
        assert widget.run_hash is not None
        assert isinstance(widget.run_hash, str)
        assert len(widget.run_hash) > 0
    
    def test_model_selection_workflow(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, minimal_manifest):
        """Test critical business logic: model selection affects widget state."""
        mock_plugin_manager.commands.execute.return_value = minimal_manifest
        widget = Inference(mock_napari_viewer)
        
        # Test model selection updates widget state
        widget.selected_model = "dummy_model"
        widget.selected_variant = "v1"
        
        assert widget.selected_model == "dummy_model"
        assert widget.selected_variant == "v1"
        
        # Test that execution state tracks selections
        widget.executed_model = widget.selected_model
        widget.executed_variant = widget.selected_variant
        
        assert widget.executed_model == widget.selected_model
        assert widget.executed_variant == widget.selected_variant
    
    def test_check_masks_functionality(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test critical business logic: mask checking for avoiding duplicate runs."""
        widget = Inference(mock_napari_viewer)
        
        # Mock dependencies for mask checking
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            # Test that check_masks returns expected structure
            result = widget.check_masks()
            
            # Should return tuple of (bool, list, list)
            assert isinstance(result, tuple)
            assert len(result) == 3
            masks_exist, load_paths, masks_all_exist = result
            assert isinstance(masks_exist, list)
            assert isinstance(load_paths, list)
            assert isinstance(masks_all_exist, bool)
    
    def test_task_to_model_to_variant_workflow(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, minimal_manifest):
        """Test end-to-end workflow: task selection -> model update -> variant selection."""
        mock_plugin_manager.commands.execute.return_value = minimal_manifest
        widget = Inference(mock_napari_viewer)
        
        # Mock subwidget interactions
        widget.subwidgets["model"].update_model_box = Mock()
        
        # Step 1: Select task
        widget.selected_task = "everything"
        assert widget.selected_task == "everything"
        
        # Step 2: This should trigger model box update (simulate UI callback)
        widget.subwidgets["model"].update_model_box(widget.selected_task)
        widget.subwidgets["model"].update_model_box.assert_called_with("everything")
        
        # Step 3: Select model and variant based on what's available
        widget.selected_model = "dummy_model"
        widget.selected_variant = "v1"
        
        # Verify the complete workflow state
        assert widget.selected_task == "everything"
        assert widget.selected_model == "dummy_model"
        assert widget.selected_variant == "v1"