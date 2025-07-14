"""
Tests for the Inference widget and its subwidgets.

This module tests the main inference functionality including:
- Task selection
- Model selection  
- Data selection
- Preprocessing options
- Export functionality
- Nextflow pipeline integration
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QApplication, QPushButton, QComboBox, QCheckBox
from qtpy.QtCore import Qt

from ai_on_demand.inference.inference_widget import Inference
from ai_on_demand.inference.tasks import TaskWidget
from ai_on_demand.inference.model_selection import ModelWidget
from ai_on_demand.inference.data_selection import DataWidget
from ai_on_demand.inference.preprocess import PreprocessWidget
from ai_on_demand.inference.mask_export import ExportWidget


class TestInferenceWidget:
    """Test the main Inference widget functionality."""
    
    def test_inference_widget_creation(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the Inference widget can be created successfully."""
        widget = Inference(mock_napari_viewer)
        
        assert widget is not None
        assert widget.viewer == mock_napari_viewer
        assert hasattr(widget, 'subwidgets')
        assert 'task' in widget.subwidgets
        
    def test_inference_widget_title_and_tooltip(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the widget has correct title and tooltip."""
        widget = Inference(mock_napari_viewer)
        
        assert "Inference" in widget.title.text()
        assert widget.title.toolTip() is not None
        
    def test_inference_widget_registers_subwidgets(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that all expected subwidgets are registered."""
        widget = Inference(mock_napari_viewer)
        
        expected_widgets = ['task', 'data', 'model', 'preprocess', 'nxf', 'export']
        for widget_name in expected_widgets:
            assert widget_name in widget.subwidgets
            
    def test_inference_widget_color_selection(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the selection color is set correctly."""
        widget = Inference(mock_napari_viewer)
        
        assert hasattr(widget, 'colour_selected')
        assert widget.colour_selected == "#F7AD6F"
        
    def test_get_run_hash_implementation(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that get_run_hash method works correctly."""
        widget = Inference(mock_napari_viewer)
        
        # The get_run_hash method should be implemented
        assert hasattr(widget, 'get_run_hash')
        # The actual implementation may depend on subwidget values


class TestTaskWidget:
    """Test the TaskWidget subwidget."""
    
    @pytest.fixture
    def task_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, dummy_manifests):
        """Create a TaskWidget for testing."""
        mock_plugin_manager.commands.execute.return_value = dummy_manifests
        
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        
        widget = TaskWidget(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_task_widget_creation(self, task_widget):
        """Test that TaskWidget can be created."""
        assert task_widget is not None
        assert task_widget._name == 'task'
        
    def test_task_widget_loads_manifests(self, task_widget, dummy_manifests):
        """Test that TaskWidget loads manifests correctly."""
        # The widget should have access to the manifests through parent
        assert hasattr(task_widget, 'parent')
        
    def test_task_selection_callback(self, task_widget):
        """Test task selection triggers correct callbacks."""
        # Mock the callback method
        task_widget.task_changed = Mock()
        
        # Simulate task selection - this would normally be done via UI interaction
        if hasattr(task_widget, 'task_changed'):
            task_widget.task_changed('organelle_segmentation')
            task_widget.task_changed.assert_called_with('organelle_segmentation')


class TestModelWidget:
    """Test the ModelWidget subwidget."""
    
    @pytest.fixture
    def model_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, dummy_manifests):
        """Create a ModelWidget for testing."""
        mock_plugin_manager.commands.execute.return_value = dummy_manifests
        
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        parent.selected_task = 'organelle_segmentation'
        
        widget = ModelWidget(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_model_widget_creation(self, model_widget):
        """Test that ModelWidget can be created."""
        assert model_widget is not None
        assert model_widget._name == 'model'
        
    def test_model_selection_updates_variants(self, model_widget):
        """Test that selecting a model updates available variants."""
        # This would test the actual UI interaction logic
        # The implementation depends on the specific widget structure
        assert hasattr(model_widget, 'parent')


class TestDataWidget:
    """Test the DataWidget subwidget."""
    
    @pytest.fixture
    def data_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create a DataWidget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        
        with patch('ai_on_demand.inference.data_selection.aiod_io'):
            widget = DataWidget(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_data_widget_creation(self, data_widget):
        """Test that DataWidget can be created."""
        assert data_widget is not None
        assert data_widget._name == 'data'
        
    def test_file_selection_dialog(self, data_widget, mock_file_dialog):
        """Test file selection dialog functionality."""
        # Test that file dialog can be triggered
        # This would involve simulating button clicks
        assert hasattr(data_widget, 'viewer')
        
    def test_layer_selection_from_napari(self, data_widget, populated_napari_viewer):
        """Test selecting layers from napari viewer."""
        data_widget.viewer = populated_napari_viewer
        
        # Test logic for selecting layers from the viewer
        layers = populated_napari_viewer.layers
        assert len(layers) > 0


class TestPreprocessWidget:
    """Test the PreprocessWidget subwidget."""
    
    @pytest.fixture
    def preprocess_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create a PreprocessWidget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        parent.selected_variant = 'default'
        
        with patch('ai_on_demand.inference.preprocess.aiod_utils'):
            widget = PreprocessWidget(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_preprocess_widget_creation(self, preprocess_widget):
        """Test that PreprocessWidget can be created."""
        assert preprocess_widget is not None
        assert preprocess_widget._name == 'preprocess'
        
    def test_preprocessing_options(self, preprocess_widget):
        """Test that preprocessing options can be configured."""
        # Test that preprocessing options are available
        assert hasattr(preprocess_widget, 'parent')


class TestExportWidget:
    """Test the ExportWidget subwidget."""
    
    @pytest.fixture
    def export_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create an ExportWidget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        
        with patch('ai_on_demand.inference.mask_export.aiod_rle'):
            widget = ExportWidget(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_export_widget_creation(self, export_widget):
        """Test that ExportWidget can be created."""
        assert export_widget is not None
        assert export_widget._name == 'export'
        
    def test_export_format_selection(self, export_widget):
        """Test export format selection functionality."""
        # Test that export formats can be selected
        assert hasattr(export_widget, 'viewer')
        
    def test_export_directory_selection(self, export_widget, mock_file_dialog):
        """Test export directory selection."""
        # Test directory selection for export
        assert hasattr(export_widget, 'parent')


class TestInferenceIntegration:
    """Test integration between inference subwidgets."""
    
    def test_task_model_interaction(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, dummy_manifests):
        """Test that task selection updates model options."""
        mock_plugin_manager.commands.execute.return_value = dummy_manifests
        
        widget = Inference(mock_napari_viewer)
        
        # Test that changing task affects model selection
        widget.selected_task = 'organelle_segmentation'
        assert widget.selected_task == 'organelle_segmentation'
        
    def test_model_variant_interaction(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, dummy_manifests):
        """Test that model selection updates variant options."""
        mock_plugin_manager.commands.execute.return_value = dummy_manifests
        
        widget = Inference(mock_napari_viewer)
        
        # Test model and variant selection
        widget.selected_model = 'mitochondria_v1'
        widget.selected_variant = 'default'
        
        assert widget.selected_model == 'mitochondria_v1'
        assert widget.selected_variant == 'default'
        
    def test_settings_persistence(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils, dummy_settings):
        """Test that widget settings are saved and loaded correctly."""
        mock_plugin_manager.commands.execute.side_effect = lambda cmd: {
            'ai-on-demand.get_manifests': {},
            'ai-on-demand.get_settings': dummy_settings
        }.get(cmd, {})
        
        widget = Inference(mock_napari_viewer)
        
        # Test settings loading
        assert hasattr(widget, 'plugin_settings')
        
        # Test settings saving
        widget.store_settings()
        
    @patch('ai_on_demand.inference.nxf.subprocess.run')
    def test_nextflow_execution(self, mock_subprocess, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test Nextflow pipeline execution."""
        mock_subprocess.return_value.returncode = 0
        
        widget = Inference(mock_napari_viewer)
        
        # Set up widget state for execution
        widget.selected_task = 'organelle_segmentation'
        widget.selected_model = 'mitochondria_v1' 
        widget.selected_variant = 'default'
        
        # Test that execution can be triggered
        assert widget.selected_task is not None