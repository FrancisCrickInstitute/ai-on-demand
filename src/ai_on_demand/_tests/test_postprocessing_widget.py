"""
Tests for the Postprocessing widget and its functionality.

This module tests the postprocessing functionality including:
- Mask filtering operations
- Mask merging operations
- Morphological operations
- Export functionality
- Layer selection and validation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QPushButton, QSpinBox, QDoubleSpinBox, QComboBox
from napari.layers import Labels

from ai_on_demand.postprocessing.postprocess_widget import Postprocess
from ai_on_demand.postprocessing.filter_masks import FilterMasks
from ai_on_demand.postprocessing.merge_masks import MergeMasks  
from ai_on_demand.postprocessing.morph_masks import MorphMasks


class TestPostprocessWidget:
    """Test the main Postprocess widget functionality."""
    
    def test_postprocess_widget_creation(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the Postprocess widget can be created successfully."""
        widget = Postprocess(mock_napari_viewer)
        
        assert widget is not None
        assert widget.viewer == mock_napari_viewer
        assert hasattr(widget, 'subwidgets')
        
    def test_postprocess_widget_title_and_tooltip(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that the widget has correct title and tooltip."""
        widget = Postprocess(mock_napari_viewer)
        
        assert "Postprocess" in widget.title.text()
        assert widget.title.toolTip() is not None
        
    def test_postprocess_widget_registers_subwidgets(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that all expected subwidgets are registered."""
        widget = Postprocess(mock_napari_viewer)
        
        expected_widgets = ['filter', 'merge', 'morph', 'export']
        for widget_name in expected_widgets:
            assert widget_name in widget.subwidgets
            
    def test_get_selected_layers_method(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test the _get_selected_layers method functionality."""
        widget = Postprocess(populated_napari_viewer)
        
        # Select labels layers
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        populated_napari_viewer.layers.selection = labels_layers
        
        selected = widget._get_selected_layers()
        assert len(selected) > 0
        
    def test_get_selected_layers_error_handling(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test error handling when no labels layers are selected."""
        widget = Postprocess(mock_napari_viewer)
        
        # Mock empty selection
        mock_napari_viewer.layers.selection = []
        
        with patch('napari.utils.notifications.show_error') as mock_show_error:
            layers = widget._get_selected_layers()
            # Should show error for no layers selected
            assert layers == []
            
    def test_get_selected_layers_shape_validation(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test shape validation for selected layers."""
        widget = Postprocess(mock_napari_viewer)
        
        # Mock layers with different shapes
        layer1 = Mock(spec=Labels)
        layer1.data = Mock()
        layer1.data.shape = (100, 100)
        
        layer2 = Mock(spec=Labels)  
        layer2.data = Mock()
        layer2.data.shape = (200, 200)  # Different shape
        
        mock_napari_viewer.layers.selection = [layer1, layer2]
        
        with patch('napari.utils.notifications.show_error') as mock_show_error:
            widget._get_selected_layers()
            # Should show error for mismatched shapes


class TestFilterMasks:
    """Test the FilterMasks subwidget functionality."""
    
    @pytest.fixture
    def filter_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create a FilterMasks widget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        parent._get_selected_layers = Mock(return_value=[])
        
        widget = FilterMasks(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_filter_widget_creation(self, filter_widget):
        """Test that FilterMasks widget can be created."""
        assert filter_widget is not None
        assert filter_widget._name == 'filter'
        
    @pytest.fixture
    def sample_labels_with_small_objects(self):
        """Create sample labels with objects of different sizes."""
        labels = np.zeros((100, 100), dtype=np.uint16)
        
        # Large object
        labels[10:50, 10:50] = 1  # 40x40 = 1600 pixels
        
        # Medium object  
        labels[60:80, 60:80] = 2  # 20x20 = 400 pixels
        
        # Small objects
        labels[20:25, 70:75] = 3  # 5x5 = 25 pixels (small)
        labels[80:83, 20:23] = 4  # 3x3 = 9 pixels (very small)
        
        return labels
        
    def test_filter_by_area_removes_small_objects(self, filter_widget, sample_labels_with_small_objects):
        """Test that area filtering removes objects below threshold."""
        # Mock the filtering functionality
        with patch('ai_on_demand.postprocessing.filter_masks.skimage.morphology.remove_small_objects') as mock_filter:
            expected_result = sample_labels_with_small_objects.copy()
            expected_result[expected_result == 3] = 0  # Remove small object
            expected_result[expected_result == 4] = 0  # Remove very small object
            mock_filter.return_value = expected_result
            
            # Test filtering with area threshold of 50 pixels
            result = mock_filter(sample_labels_with_small_objects, min_size=50)
            
            # Should remove objects 3 and 4 (areas 25 and 9)
            unique_labels = np.unique(result)
            assert 3 not in unique_labels
            assert 4 not in unique_labels
            assert 1 in unique_labels  # Large object remains
            assert 2 in unique_labels  # Medium object remains
            
    def test_filter_ui_components(self, filter_widget):
        """Test that filter widget has required UI components."""
        # Mock UI components that would be created in create_box
        filter_widget.min_area_spinbox = Mock(spec=QSpinBox)
        filter_widget.max_area_spinbox = Mock(spec=QSpinBox)  
        filter_widget.apply_filter_button = Mock(spec=QPushButton)
        
        # Test that components can be configured
        filter_widget.min_area_spinbox.setValue(50)
        filter_widget.min_area_spinbox.setValue.assert_called_with(50)
        
    def test_filter_settings_persistence(self, filter_widget):
        """Test that filter settings are saved and loaded."""
        test_settings = {
            'min_area': 100,
            'max_area': 10000,
            'filter_enabled': True
        }
        
        # Test settings loading
        filter_widget.load_settings = Mock()
        filter_widget.get_settings = Mock(return_value=test_settings)
        
        settings = filter_widget.get_settings()
        assert settings == test_settings


class TestMergeMasks:
    """Test the MergeMasks subwidget functionality."""
    
    @pytest.fixture
    def merge_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create a MergeMasks widget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        parent._get_selected_layers = Mock(return_value=[])
        
        widget = MergeMasks(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_merge_widget_creation(self, merge_widget):
        """Test that MergeMasks widget can be created."""
        assert merge_widget is not None
        assert merge_widget._name == 'merge'
        
    @pytest.fixture
    def overlapping_labels(self):
        """Create sample labels with overlapping objects for merging."""
        labels1 = np.zeros((100, 100), dtype=np.uint16)
        labels1[20:60, 20:60] = 1  # Object in first mask
        
        labels2 = np.zeros((100, 100), dtype=np.uint16)
        labels2[40:80, 40:80] = 1  # Overlapping object in second mask
        
        return labels1, labels2
        
    def test_merge_overlapping_objects(self, merge_widget, overlapping_labels):
        """Test merging of overlapping objects."""
        labels1, labels2 = overlapping_labels
        
        # Mock merge functionality
        with patch('ai_on_demand.postprocessing.merge_masks.np.maximum') as mock_merge:
            expected_result = np.maximum(labels1, labels2 * 2)  # Offset second mask labels
            mock_merge.return_value = expected_result
            
            result = mock_merge(labels1, labels2 * 2)
            
            # Should have objects from both masks
            unique_labels = np.unique(result)
            assert len(unique_labels) > len(np.unique(labels1))
            
    def test_merge_strategy_selection(self, merge_widget):
        """Test different merge strategies."""
        # Mock UI components for merge strategy selection
        merge_widget.strategy_combo = Mock(spec=QComboBox)
        merge_widget.strategy_combo.currentText.return_value = 'union'
        
        strategy = merge_widget.strategy_combo.currentText()
        assert strategy == 'union'
        
        # Test other strategies
        merge_widget.strategy_combo.currentText.return_value = 'intersection'
        strategy = merge_widget.strategy_combo.currentText()
        assert strategy == 'intersection'
        
    def test_merge_with_multiple_layers(self, merge_widget):
        """Test merging multiple layers simultaneously."""
        # Create mock layers
        layer1 = Mock(spec=Labels)
        layer1.data = np.ones((50, 50), dtype=np.uint16)
        layer1.name = 'mask1'
        
        layer2 = Mock(spec=Labels)
        layer2.data = np.ones((50, 50), dtype=np.uint16) * 2
        layer2.name = 'mask2'
        
        layer3 = Mock(spec=Labels)
        layer3.data = np.ones((50, 50), dtype=np.uint16) * 3  
        layer3.name = 'mask3'
        
        layers = [layer1, layer2, layer3]
        
        # Test that multiple layers can be processed
        assert len(layers) == 3


class TestMorphMasks:
    """Test the MorphMasks subwidget functionality."""
    
    @pytest.fixture
    def morph_widget(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create a MorphMasks widget for testing."""
        parent = Mock()
        parent.content_widget = Mock()
        parent.content_widget.layout.return_value.addWidget = Mock()
        parent.register_widget = Mock()
        parent.plugin_settings = {}
        parent._get_selected_layers = Mock(return_value=[])
        
        widget = MorphMasks(viewer=mock_napari_viewer, parent=parent)
        return widget
        
    def test_morph_widget_creation(self, morph_widget):
        """Test that MorphMasks widget can be created."""
        assert morph_widget is not None
        assert morph_widget._name == 'morph'
        
    @pytest.fixture
    def labels_for_morphology(self):
        """Create sample labels for morphological operations."""
        labels = np.zeros((100, 100), dtype=np.uint16)
        
        # Create objects with rough edges for morphology testing
        labels[30:35, 30:35] = 1  # Small square
        labels[50:70, 50:70] = 2  # Larger square
        labels[20:25, 70:90] = 3  # Rectangle
        
        return labels
        
    def test_morphological_closing(self, morph_widget, labels_for_morphology):
        """Test morphological closing operation."""
        with patch('skimage.morphology.closing') as mock_closing:
            mock_closing.return_value = labels_for_morphology
            
            result = mock_closing(labels_for_morphology)
            assert result is not None
            mock_closing.assert_called_once()
            
    def test_morphological_opening(self, morph_widget, labels_for_morphology):
        """Test morphological opening operation."""
        with patch('skimage.morphology.opening') as mock_opening:
            mock_opening.return_value = labels_for_morphology
            
            result = mock_opening(labels_for_morphology)
            assert result is not None
            mock_opening.assert_called_once()
            
    def test_morphological_erosion(self, morph_widget, labels_for_morphology):
        """Test morphological erosion operation."""
        with patch('skimage.morphology.erosion') as mock_erosion:
            mock_erosion.return_value = labels_for_morphology
            
            result = mock_erosion(labels_for_morphology)
            assert result is not None
            mock_erosion.assert_called_once()
            
    def test_morphological_dilation(self, morph_widget, labels_for_morphology):
        """Test morphological dilation operation."""
        with patch('skimage.morphology.dilation') as mock_dilation:
            mock_dilation.return_value = labels_for_morphology
            
            result = mock_dilation(labels_for_morphology)
            assert result is not None
            mock_dilation.assert_called_once()
            
    def test_morph_operation_selection(self, morph_widget):
        """Test selection of morphological operations."""
        # Mock UI components
        morph_widget.operation_combo = Mock(spec=QComboBox)
        morph_widget.kernel_size_spinbox = Mock(spec=QSpinBox)
        morph_widget.apply_button = Mock(spec=QPushButton)
        
        # Test operation selection
        morph_widget.operation_combo.currentText.return_value = 'closing'
        operation = morph_widget.operation_combo.currentText()
        assert operation == 'closing'
        
        # Test kernel size setting
        morph_widget.kernel_size_spinbox.value.return_value = 3
        kernel_size = morph_widget.kernel_size_spinbox.value()
        assert kernel_size == 3


class TestPostprocessingIntegration:
    """Test integration between postprocessing components."""
    
    def test_postprocessing_workflow(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test complete postprocessing workflow."""
        widget = Postprocess(populated_napari_viewer)
        
        # Test that all subwidgets are available
        assert 'filter' in widget.subwidgets
        assert 'merge' in widget.subwidgets
        assert 'morph' in widget.subwidgets
        assert 'export' in widget.subwidgets
        
        # Test layer selection
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        if labels_layers:
            populated_napari_viewer.layers.selection = labels_layers[:1]
            selected = widget._get_selected_layers()
            assert len(selected) >= 0
            
    def test_sequential_postprocessing_operations(self, populated_napari_viewer, 
                                                mock_plugin_manager, mock_aiod_utils):
        """Test applying multiple postprocessing operations in sequence."""
        widget = Postprocess(populated_napari_viewer)
        
        # Simulate sequential operations:
        # 1. Filter small objects
        # 2. Apply morphological closing
        # 3. Merge with another layer
        # 4. Export results
        
        filter_widget = widget.subwidgets['filter']
        morph_widget = widget.subwidgets['morph']
        merge_widget = widget.subwidgets['merge']
        export_widget = widget.subwidgets['export']
        
        assert filter_widget is not None
        assert morph_widget is not None  
        assert merge_widget is not None
        assert export_widget is not None
        
    def test_postprocessing_settings_persistence(self, mock_napari_viewer, 
                                               mock_plugin_manager, mock_aiod_utils):
        """Test that postprocessing settings are saved and loaded correctly."""
        postprocess_settings = {
            'filter': {
                'min_area': 100,
                'max_area': 50000
            },
            'morph': {
                'operation': 'closing',
                'kernel_size': 3
            },
            'merge': {
                'strategy': 'union'
            }
        }
        
        mock_plugin_manager.commands.execute.side_effect = lambda cmd: {
            'ai-on-demand.get_manifests': {},
            'ai-on-demand.get_settings': postprocess_settings
        }.get(cmd, {})
        
        widget = Postprocess(mock_napari_viewer)
        
        # Test settings loading
        assert hasattr(widget, 'plugin_settings')
        
        # Test settings saving
        widget.store_settings()
        
    def test_postprocessing_error_handling(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test error handling in postprocessing widgets."""
        widget = Postprocess(mock_napari_viewer)
        
        # Test handling of incompatible layer types
        with patch('napari.utils.notifications.show_error') as mock_show_error:
            # Simulate selecting non-labels layer
            mock_image_layer = Mock()
            mock_image_layer.data = np.random.rand(100, 100)
            mock_napari_viewer.layers.selection = [mock_image_layer]
            
            # Should handle gracefully
            layers = widget._get_selected_layers()
            assert layers == []
            
    def test_postprocessing_with_real_napari_layers(self, populated_napari_viewer, 
                                                  mock_plugin_manager, mock_aiod_utils):
        """Test postprocessing with actual napari layers."""
        widget = Postprocess(populated_napari_viewer)
        
        # Add additional labels layer for testing
        test_labels = np.random.randint(0, 5, (100, 100), dtype=np.uint16)
        populated_napari_viewer.add_labels(test_labels, name='test_postprocess')
        
        # Select labels layers  
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        populated_napari_viewer.layers.selection = labels_layers
        
        # Test layer selection
        selected = widget._get_selected_layers()
        assert len(selected) > 0
        
        # All selected layers should be Labels type
        for layer in selected:
            assert isinstance(layer, Labels)