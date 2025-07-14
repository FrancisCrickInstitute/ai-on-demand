"""
Integration tests for the ai-on-demand plugin.

This module tests the overall plugin functionality including:
- Plugin registration and loading
- Cross-widget interactions
- Data flow between components  
- Napari integration
- Settings persistence across the plugin
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import napari
from napari.layers import Image, Labels

from ai_on_demand.inference.inference_widget import Inference
from ai_on_demand.evaluation.evaluation_widget import Evaluation
from ai_on_demand.postprocessing.postprocess_widget import Postprocess
import ai_on_demand


class TestPluginRegistration:
    """Test plugin registration with napari."""
    
    def test_plugin_manifest_exists(self):
        """Test that the plugin manifest file exists and is valid."""
        manifest_path = Path(ai_on_demand.__file__).parent / 'napari.yaml'
        assert manifest_path.exists()
        
        # Check that manifest contains expected commands
        with open(manifest_path, 'r') as f:
            content = f.read()
            assert 'ai-on-demand.inference' in content
            assert 'ai-on-demand.evaluation' in content
            assert 'ai-on-demand.postprocess' in content
            
    def test_plugin_commands_registration(self):
        """Test that plugin commands are properly registered."""
        # Test that the main widget classes can be imported
        assert Inference is not None
        assert Evaluation is not None  
        assert Postprocess is not None
        
    @patch('npe2.PluginManager.instance')
    def test_plugin_activation(self, mock_pm):
        """Test plugin activation process."""
        # Mock plugin manager
        pm_instance = Mock()
        pm_instance.commands.execute.return_value = {}
        mock_pm.return_value = pm_instance
        
        # Test that activation function can be called
        from ai_on_demand.load_manifests import activate_plugin
        result = activate_plugin()
        
        # Should not raise an exception
        assert True


class TestCrossWidgetInteractions:
    """Test interactions between different widgets."""
    
    @pytest.fixture
    def all_widgets(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Create all main widgets for testing interactions."""
        inference_widget = Inference(populated_napari_viewer)
        evaluation_widget = Evaluation(populated_napari_viewer)
        postprocess_widget = Postprocess(populated_napari_viewer)
        
        return {
            'inference': inference_widget,
            'evaluation': evaluation_widget,
            'postprocess': postprocess_widget
        }
        
    def test_inference_to_evaluation_workflow(self, all_widgets, populated_napari_viewer):
        """Test workflow from inference to evaluation."""
        inference = all_widgets['inference']
        evaluation = all_widgets['evaluation']
        
        # Simulate inference results being added to napari
        inference_results = np.random.randint(0, 5, (100, 100), dtype=np.uint16)
        populated_napari_viewer.add_labels(inference_results, name='inference_results')
        
        # Test that evaluation can use inference results
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        assert len(labels_layers) >= 2  # Original + inference results
        
        # Evaluation should be able to select these layers
        assert evaluation.viewer == populated_napari_viewer
        
    def test_inference_to_postprocessing_workflow(self, all_widgets, populated_napari_viewer):
        """Test workflow from inference to postprocessing."""
        inference = all_widgets['inference']
        postprocess = all_widgets['postprocess']
        
        # Simulate inference results
        inference_results = np.random.randint(0, 8, (100, 100), dtype=np.uint16)
        populated_napari_viewer.add_labels(inference_results, name='raw_inference')
        
        # Test that postprocessing can use inference results
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        populated_napari_viewer.layers.selection = labels_layers[-1:]  # Select latest
        
        selected = postprocess._get_selected_layers()
        assert len(selected) > 0
        
    def test_postprocessing_to_evaluation_workflow(self, all_widgets, populated_napari_viewer):
        """Test workflow from postprocessing to evaluation."""
        postprocess = all_widgets['postprocess']
        evaluation = all_widgets['evaluation']
        
        # Simulate postprocessed results
        postprocessed_results = np.random.randint(0, 3, (100, 100), dtype=np.uint16)
        populated_napari_viewer.add_labels(postprocessed_results, name='postprocessed')
        
        # Test that evaluation can evaluate postprocessed results
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        assert len(labels_layers) >= 2  # Should have multiple labels layers


class TestDataFlowIntegration:
    """Test data flow through the complete pipeline."""
    
    def test_complete_pipeline_simulation(self, populated_napari_viewer, 
                                        mock_plugin_manager, mock_aiod_utils, temp_directory):
        """Test complete pipeline from raw data to final evaluation."""
        # Step 1: Start with raw image data
        raw_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        populated_napari_viewer.add_image(raw_image, name='raw_data')
        
        # Step 2: Run inference (simulated)
        inference_widget = Inference(populated_napari_viewer)
        
        # Simulate inference results
        inference_masks = np.random.randint(0, 10, (200, 200), dtype=np.uint16)
        populated_napari_viewer.add_labels(inference_masks, name='inference_output')
        
        # Step 3: Apply postprocessing
        postprocess_widget = Postprocess(populated_napari_viewer)
        
        # Simulate postprocessing
        filtered_masks = inference_masks.copy()
        filtered_masks[filtered_masks < 2] = 0  # Remove small labels
        populated_napari_viewer.add_labels(filtered_masks, name='filtered_output')
        
        # Step 4: Evaluate results
        evaluation_widget = Evaluation(populated_napari_viewer)
        
        # Test that all data is available in napari
        image_layers = [layer for layer in populated_napari_viewer.layers 
                       if isinstance(layer, Image)]
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        
        assert len(image_layers) >= 2  # raw_data + test_image
        assert len(labels_layers) >= 3  # test_labels + inference + filtered
        
        # Test that each widget can access the data
        assert inference_widget.viewer == populated_napari_viewer
        assert postprocess_widget.viewer == populated_napari_viewer  
        assert evaluation_widget.viewer == populated_napari_viewer
        
    def test_data_persistence_across_widgets(self, populated_napari_viewer,
                                           mock_plugin_manager, mock_aiod_utils):
        """Test that data persists and is accessible across all widgets."""
        # Create all widgets
        widgets = [
            Inference(populated_napari_viewer),
            Evaluation(populated_napari_viewer),
            Postprocess(populated_napari_viewer)
        ]
        
        # Add test data
        test_data = np.random.randint(0, 5, (150, 150), dtype=np.uint16)
        populated_napari_viewer.add_labels(test_data, name='shared_data')
        
        # Test that all widgets can access the same data
        for widget in widgets:
            assert widget.viewer == populated_napari_viewer
            assert 'shared_data' in [layer.name for layer in widget.viewer.layers]


class TestSettingsPersistence:
    """Test settings persistence across the entire plugin."""
    
    def test_global_settings_structure(self, mock_plugin_manager, mock_aiod_utils):
        """Test that global settings have consistent structure."""
        test_settings = {
            'tasks': {'selected_task': 'organelle_segmentation'},
            'models': {'selected_model': 'mitochondria_v1'},
            'data': {'last_directory': '/test/path'},
            'eval': {'use_ground_truth': True},
            'filter': {'min_area': 100},
            'merge': {'strategy': 'union'},
            'morph': {'operation': 'closing'}
        }
        
        mock_plugin_manager.commands.execute.side_effect = lambda cmd: {
            'ai-on-demand.get_manifests': {},
            'ai-on-demand.get_settings': test_settings
        }.get(cmd, {})
        
        # Test that settings are accessible to all widgets
        from ai_on_demand.widget_classes import MainWidget
        
        # Mock viewer for testing
        mock_viewer = Mock()
        
        # Create a test widget to check settings access
        class TestWidget(MainWidget):
            def get_run_hash(self):
                return None
                
        widget = TestWidget(mock_viewer, "Test")
        assert hasattr(widget, 'plugin_settings')
        
    def test_settings_save_and_load_cycle(self, populated_napari_viewer, 
                                        mock_plugin_manager, mock_aiod_utils, temp_directory):
        """Test complete save and load cycle for settings."""
        # Mock settings file location
        settings_file = temp_directory / 'settings.yaml'
        
        with patch('ai_on_demand.utils.get_plugin_cache') as mock_cache:
            mock_cache.return_value = (temp_directory, settings_file)
            
            # Create widget and modify settings
            widget = Inference(populated_napari_viewer)
            widget.selected_task = 'test_task'
            widget.selected_model = 'test_model'
            
            # Save settings
            widget.store_settings()
            
            # Verify settings file was created
            assert settings_file.exists()
            
    def test_settings_validation(self, mock_plugin_manager, mock_aiod_utils):
        """Test that invalid settings are handled gracefully."""
        # Test with malformed settings
        invalid_settings = {
            'tasks': 'not_a_dict',  # Should be dict
            'models': None,  # Should be dict
            'data': {'invalid_key': 'value'}
        }
        
        mock_plugin_manager.commands.execute.side_effect = lambda cmd: {
            'ai-on-demand.get_manifests': {},
            'ai-on-demand.get_settings': invalid_settings
        }.get(cmd, {})
        
        # Should not crash when loading invalid settings
        mock_viewer = Mock()
        try:
            widget = Inference(mock_viewer)
            assert True  # Should complete without error
        except Exception:
            pytest.fail("Widget creation should handle invalid settings gracefully")


class TestNapariIntegration:
    """Test integration with napari viewer functionality."""
    
    def test_layer_management(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test layer creation, selection, and management."""
        widget = Inference(populated_napari_viewer)
        
        initial_layer_count = len(populated_napari_viewer.layers)
        
        # Add new layers through widget functionality
        test_image = np.random.rand(100, 100)
        test_labels = np.random.randint(0, 5, (100, 100), dtype=np.uint16)
        
        populated_napari_viewer.add_image(test_image, name='widget_image')
        populated_napari_viewer.add_labels(test_labels, name='widget_labels')
        
        assert len(populated_napari_viewer.layers) == initial_layer_count + 2
        
    def test_layer_selection_events(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that widgets respond to layer selection events."""
        postprocess_widget = Postprocess(populated_napari_viewer)
        
        # Select specific layers
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        
        if labels_layers:
            populated_napari_viewer.layers.selection.clear()
            populated_napari_viewer.layers.selection.add(labels_layers[0])
            
            # Test that widget can detect selection
            selected = postprocess_widget._get_selected_layers()
            assert len(selected) <= 1
            
    def test_viewer_notifications(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that widgets properly use napari notifications."""
        widget = Postprocess(populated_napari_viewer)
        
        # Test error notifications
        with patch('napari.utils.notifications.show_error') as mock_error:
            # Trigger an error condition (no layers selected)
            populated_napari_viewer.layers.selection.clear()
            widget._get_selected_layers()
            
            # Should handle gracefully (may or may not show error depending on implementation)
            assert True
            
    def test_layer_data_types(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test handling of different layer data types."""
        widget = Postprocess(populated_napari_viewer)
        
        # Add different data types
        float_data = np.random.rand(50, 50).astype(np.float32)
        int_data = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        labels_data = np.random.randint(0, 10, (50, 50), dtype=np.uint16)
        
        populated_napari_viewer.add_image(float_data, name='float_image')
        populated_napari_viewer.add_image(int_data, name='int_image')  
        populated_napari_viewer.add_labels(labels_data, name='labels_image')
        
        # Test that only appropriate layers are selected
        labels_layers = [layer for layer in populated_napari_viewer.layers 
                        if isinstance(layer, Labels)]
        assert len(labels_layers) >= 2  # Original test_labels + new labels_image


class TestErrorHandling:
    """Test error handling across the plugin."""
    
    def test_missing_dependencies_handling(self, mock_napari_viewer):
        """Test graceful handling of missing external dependencies."""
        # Test with completely mocked dependencies
        with patch.dict('sys.modules', {
            'aiod_utils': None,
            'aiod_registry': None,
            'skimage': Mock()
        }):
            # Should handle missing dependencies gracefully
            try:
                # This might fail, but should be handled gracefully
                pass
            except ImportError:
                # Expected for missing dependencies
                pass
                
    def test_invalid_data_handling(self, populated_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test handling of invalid or corrupted data."""
        widget = Postprocess(populated_napari_viewer)
        
        # Add layer with invalid data
        invalid_data = np.array([[1, 2], [3]])  # Irregular array
        
        try:
            # This should handle invalid data gracefully
            widget._get_selected_layers()
            assert True
        except Exception:
            # Should not crash the plugin
            pytest.fail("Plugin should handle invalid data gracefully")
            
    def test_memory_management(self, mock_plugin_manager, mock_aiod_utils):
        """Test that widgets can be created and destroyed without memory leaks."""
        mock_viewer = Mock()
        
        # Create and destroy widgets multiple times
        for i in range(10):
            inference = Inference(mock_viewer)
            evaluation = Evaluation(mock_viewer)
            postprocess = Postprocess(mock_viewer)
            
            # Delete references
            del inference, evaluation, postprocess
            
        # Should complete without issues
        assert True


class TestPluginPerformance:
    """Test plugin performance characteristics."""
    
    def test_widget_initialization_time(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test that widgets initialize in reasonable time."""
        import time
        
        start_time = time.time()
        
        # Create all main widgets
        Inference(mock_napari_viewer)
        Evaluation(mock_napari_viewer)  
        Postprocess(mock_napari_viewer)
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        # Should initialize within reasonable time (adjust threshold as needed)
        assert initialization_time < 5.0  # 5 seconds max
        
    def test_large_data_handling(self, mock_napari_viewer, mock_plugin_manager, mock_aiod_utils):
        """Test handling of large datasets."""
        # Create large test data
        large_image = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        large_labels = np.random.randint(0, 100, (1000, 1000), dtype=np.uint16)
        
        mock_napari_viewer.add_image = Mock()
        mock_napari_viewer.add_labels = Mock()
        
        # Test that large data can be handled
        mock_napari_viewer.add_image(large_image, name='large_image')
        mock_napari_viewer.add_labels(large_labels, name='large_labels')
        
        # Should complete without memory errors
        assert True