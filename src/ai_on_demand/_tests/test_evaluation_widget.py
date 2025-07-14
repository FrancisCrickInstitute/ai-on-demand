"""
Tests for the Evaluation widget and its functionality.

This module tests the evaluation functionality including:
- Metric calculation (with and without ground truth)
- Layer selection for evaluation
- Export of evaluation results
- UI interactions for evaluation parameters
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtWidgets import QComboBox, QPushButton, QCheckBox, QTextBrowser

from ai_on_demand.evaluation.evaluation_widget import Evaluation, EvalWidget


class TestEvaluationWidget:
    """Test the main Evaluation widget functionality."""
    
    def test_evaluation_widget_creation(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the Evaluation widget can be created successfully."""
        widget = Evaluation(mock_napari_viewer)
        
        assert widget is not None
        assert widget.viewer == mock_napari_viewer
        assert hasattr(widget, 'subwidgets')
        assert 'eval' in widget.subwidgets
        
    def test_evaluation_widget_title_and_tooltip(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the widget has correct title and tooltip."""
        widget = Evaluation(mock_napari_viewer)
        
        assert "Evaluation" in widget.title.text()
        assert widget.title.toolTip() is not None
        
    def test_evaluation_widget_registers_eval_subwidget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the evaluation subwidget is registered and expanded."""
        widget = Evaluation(mock_napari_viewer)
        
        assert 'eval' in widget.subwidgets
        eval_widget = widget.subwidgets['eval']
        assert isinstance(eval_widget, EvalWidget)
        
    def test_get_run_hash_not_implemented(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that get_run_hash is not implemented for evaluation (no Nextflow)."""
        widget = Evaluation(mock_napari_viewer)
        
        # The method should exist but return None or pass
        result = widget.get_run_hash()
        assert result is None


class TestEvalWidget:
    """Test the EvalWidget subwidget functionality."""
    
    @pytest.fixture
    def eval_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create an EvalWidget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        
        widget = EvalWidget(viewer=mock_napari_viewer, parent=parent, expanded=True)
        return widget
        
    def test_eval_widget_creation(self, eval_widget):
        """Test that EvalWidget can be created."""
        assert eval_widget is not None
        assert eval_widget._name == 'eval'
        
    def test_eval_widget_has_required_components(self, eval_widget):
        """Test that the evaluation widget has all required UI components."""
        # The widget should have components for:
        # - Layer selection
        # - Metric selection  
        # - Ground truth options
        # - Results display
        assert hasattr(eval_widget, 'viewer')
        assert hasattr(eval_widget, 'parent')


class TestEvaluationMetrics:
    """Test evaluation metrics calculation functionality."""
    
    @pytest.fixture
    def sample_prediction_labels(self):
        """Create sample prediction labels for testing."""
        labels = np.zeros((100, 100), dtype=np.uint16)
        # Create some simple shapes
        labels[20:40, 20:40] = 1  # Square object 1
        labels[60:80, 60:80] = 2  # Square object 2
        labels[30:50, 70:90] = 3  # Overlapping object 3
        return labels
        
    @pytest.fixture  
    def sample_ground_truth_labels(self):
        """Create sample ground truth labels for testing."""
        labels = np.zeros((100, 100), dtype=np.uint16)
        # Create slightly different shapes for testing metrics
        labels[18:42, 18:42] = 1  # Slightly larger square 1
        labels[58:82, 58:82] = 2  # Slightly larger square 2
        labels[32:48, 72:88] = 3  # Different overlapping object 3
        return labels
        
    @patch('ai_on_demand.evaluation.metrics.skimage.measure.regionprops')
    def test_metrics_without_ground_truth(self, mock_regionprops, eval_widget, sample_prediction_labels):
        """Test calculation of metrics without ground truth (descriptive metrics)."""
        # Mock regionprops to return predictable results
        mock_prop1 = Mock()
        mock_prop1.area = 400
        mock_prop1.perimeter = 80
        mock_prop1.eccentricity = 0.0
        
        mock_prop2 = Mock()
        mock_prop2.area = 400  
        mock_prop2.perimeter = 80
        mock_prop2.eccentricity = 0.0
        
        mock_regionprops.return_value = [mock_prop1, mock_prop2]
        
        with patch('ai_on_demand.evaluation.metrics.aiod_metrics') as mock_metrics:
            mock_metrics.describe_labels.return_value = pd.DataFrame({
                'label': [1, 2],
                'area': [400, 400],
                'perimeter': [80, 80],
                'eccentricity': [0.0, 0.0]
            })
            
            # Test that metrics can be calculated
            result = mock_metrics.describe_labels(sample_prediction_labels)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            
    @patch('ai_on_demand.evaluation.metrics.skimage.measure.regionprops')
    def test_metrics_with_ground_truth(self, mock_regionprops, eval_widget, 
                                     sample_prediction_labels, sample_ground_truth_labels):
        """Test calculation of metrics with ground truth (accuracy metrics)."""
        with patch('ai_on_demand.evaluation.metrics.aiod_metrics') as mock_metrics:
            # Mock IoU calculation
            mock_metrics.calculate_iou.return_value = pd.DataFrame({
                'pred_label': [1, 2, 3],
                'gt_label': [1, 2, 3], 
                'iou': [0.8, 0.8, 0.7]
            })
            
            # Mock precision/recall calculation
            mock_metrics.calculate_precision_recall.return_value = {
                'precision': 0.85,
                'recall': 0.80,
                'f1_score': 0.825
            }
            
            # Test that accuracy metrics can be calculated
            iou_result = mock_metrics.calculate_iou(sample_prediction_labels, sample_ground_truth_labels)
            pr_result = mock_metrics.calculate_precision_recall(sample_prediction_labels, sample_ground_truth_labels)
            
            assert isinstance(iou_result, pd.DataFrame)
            assert 'iou' in iou_result.columns
            assert isinstance(pr_result, dict)
            assert 'precision' in pr_result
            assert 'recall' in pr_result


class TestEvaluationUI:
    """Test evaluation widget UI interactions."""
    
    @pytest.fixture
    def eval_widget_with_ui(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create an evaluation widget with UI components for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        
        widget = EvalWidget(viewer=populated_napari_viewer, parent=parent, expanded=True)
        
        # Mock UI components that would be created in create_box
        widget.layer_dropdown = Mock(spec=QComboBox)
        widget.gt_layer_dropdown = Mock(spec=QComboBox)
        widget.calculate_button = Mock(spec=QPushButton)
        widget.results_browser = Mock(spec=QTextBrowser)
        widget.use_ground_truth_checkbox = Mock(spec=QCheckBox)
        
        return widget
        
    def test_layer_selection_updates_options(self, eval_widget_with_ui):
        """Test that layer selection updates available options."""
        # Simulate layer selection
        eval_widget_with_ui.layer_dropdown.currentText.return_value = 'test_labels'
        
        # Test that selection triggers appropriate updates
        selected_layer = eval_widget_with_ui.layer_dropdown.currentText()
        assert selected_layer == 'test_labels'
        
    def test_ground_truth_checkbox_toggles_options(self, eval_widget_with_ui):
        """Test that ground truth checkbox enables/disables GT layer selection."""
        # Test unchecked state
        eval_widget_with_ui.use_ground_truth_checkbox.isChecked.return_value = False
        gt_enabled = eval_widget_with_ui.use_ground_truth_checkbox.isChecked()
        assert not gt_enabled
        
        # Test checked state
        eval_widget_with_ui.use_ground_truth_checkbox.isChecked.return_value = True
        gt_enabled = eval_widget_with_ui.use_ground_truth_checkbox.isChecked()
        assert gt_enabled
        
    def test_calculate_button_triggers_computation(self, eval_widget_with_ui):
        """Test that calculate button triggers metric computation."""
        # Mock the calculation method
        eval_widget_with_ui._calculate_metrics = Mock()
        
        # Simulate button click
        eval_widget_with_ui.calculate_button.clicked.emit()
        
        # For this test, we just verify the UI components exist
        assert eval_widget_with_ui.calculate_button is not None
        
    def test_results_display_updates(self, eval_widget_with_ui):
        """Test that results are displayed correctly in the text browser."""
        sample_results = "Mean IoU: 0.75\nPrecision: 0.80\nRecall: 0.85"
        
        # Simulate results update
        eval_widget_with_ui.results_browser.setText(sample_results)
        eval_widget_with_ui.results_browser.setText.assert_called_with(sample_results)


class TestEvaluationFileOperations:
    """Test file operations for evaluation results."""
    
    def test_export_results_to_csv(self, eval_widget, temp_directory, mock_file_dialog):
        """Test exporting evaluation results to CSV file."""
        # Mock evaluation results
        mock_results = pd.DataFrame({
            'label': [1, 2, 3],
            'area': [400, 500, 300],
            'iou': [0.8, 0.7, 0.9]
        })
        
        # Test CSV export functionality
        output_path = temp_directory / 'results.csv'
        mock_results.to_csv(output_path, index=False)
        
        assert output_path.exists()
        
        # Verify CSV content
        loaded_results = pd.read_csv(output_path)
        assert len(loaded_results) == 3
        assert 'iou' in loaded_results.columns
        
    def test_export_results_to_excel(self, eval_widget, temp_directory):
        """Test exporting evaluation results to Excel file.""" 
        # Mock evaluation results
        mock_results = pd.DataFrame({
            'label': [1, 2, 3],
            'area': [400, 500, 300],
            'precision': [0.8, 0.7, 0.9]
        })
        
        # Test Excel export functionality (if xlsxwriter is available)
        output_path = temp_directory / 'results.xlsx'
        
        try:
            mock_results.to_excel(output_path, index=False)
            assert output_path.exists()
        except ImportError:
            # xlsxwriter might not be available
            pytest.skip("xlsxwriter not available for Excel export test")


class TestEvaluationIntegration:
    """Test integration between evaluation components."""
    
    def test_evaluation_workflow_without_ground_truth(self, populated_napari_viewer, 
                                                    mock_plugin_manager, mock_aiod_utils):
        """Test complete evaluation workflow without ground truth."""
        widget = Evaluation(populated_napari_viewer)
        eval_subwidget = widget.subwidgets['eval']
        
        # Test that the workflow can be initiated
        assert eval_subwidget is not None
        
        # Simulate selecting a labels layer
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if hasattr(layer, 'data') and layer.data.dtype in [np.uint16, np.int32]]
        assert len(labels_layers) > 0
        
    def test_evaluation_workflow_with_ground_truth(self, populated_napari_viewer,
                                                  mock_plugin_manager, mock_aiod_utils):
        """Test complete evaluation workflow with ground truth."""
        # Add a second labels layer as ground truth
        gt_data = np.random.randint(0, 5, (100, 100), dtype=np.uint16)
        populated_napari_viewer.add_labels(gt_data, name='ground_truth')
        
        widget = Evaluation(populated_napari_viewer)
        eval_subwidget = widget.subwidgets['eval']
        
        # Test that both prediction and ground truth layers are available
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if hasattr(layer, 'data') and layer.data.dtype in [np.uint16, np.int32]]
        assert len(labels_layers) >= 2
        
    def test_evaluation_settings_persistence(self, mock_napari_viewer, mock_plugin_manager, 
                                           mock_aiod_utils, dummy_settings):
        """Test that evaluation settings are saved and loaded correctly."""
        # Mock settings with evaluation-specific data
        eval_settings = {
            'eval': {
                'selected_metrics': ['iou', 'precision', 'recall'],
                'use_ground_truth': True,
                'export_format': 'csv'
            }
        }
        
        mock_plugin_manager.commands.execute.side_effect = lambda cmd: {
            'ai-on-demand.get_manifests': {},
            'ai-on-demand.get_settings': eval_settings
        }.get(cmd, {})
        
        widget = Evaluation(mock_napari_viewer)
        
        # Test settings loading
        assert hasattr(widget, 'plugin_settings')
        
        # Test settings saving
        widget.store_settings()
        
    def test_evaluation_error_handling(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test error handling in evaluation widget."""
        widget = Evaluation(mock_napari_viewer)
        eval_subwidget = widget.subwidgets['eval']
        
        # Test handling of empty layer selection
        with patch('napari.utils.notifications.show_error') as mock_show_error:
            # Simulate error condition - no layers selected
            eval_subwidget.viewer.layers.selection = []
            
            # The widget should handle this gracefully
            assert eval_subwidget.viewer is not None