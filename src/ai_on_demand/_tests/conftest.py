"""
Pytest configuration file for ai-on-demand plugin tests.

This file contains common fixtures used across all test modules.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import napari
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any


@pytest.fixture
def mock_napari_viewer():
    """Create a mock napari viewer for testing."""
    viewer = Mock(spec=napari.Viewer)
    
    # Create mock layers list with selection attribute
    mock_layers = Mock()
    mock_layers.__iter__ = Mock(return_value=iter([]))  # Empty iterator
    mock_layers.__len__ = Mock(return_value=0)
    mock_layers.__bool__ = Mock(return_value=False)  # Empty layers list
    
    # Create mock selection with events
    mock_selection = Mock()
    mock_events = Mock()
    mock_events.changed = Mock()
    mock_events.changed.connect = Mock()
    mock_selection.events = mock_events
    mock_selection.__iter__ = Mock(return_value=iter([]))
    mock_selection.__len__ = Mock(return_value=0)
    
    mock_layers.selection = mock_selection
    
    viewer.layers = mock_layers
    viewer.add_image = Mock()
    viewer.add_labels = Mock()
    return viewer


@pytest.fixture 
def dummy_manifests():
    """Create dummy manifests for testing plugin functionality."""
    return {
        'organelle_segmentation': {
            'display_name': 'Organelle Segmentation',
            'description': 'Test organelle segmentation model',
            'models': {
                'mitochondria_v1': {
                    'display_name': 'Mitochondria v1',
                    'description': 'Test mitochondria model',
                    'variants': {
                        'default': {
                            'display_name': 'Default',
                            'preprocessing': ['normalize', 'resize']
                        }
                    }
                }
            }
        },
        'cell_segmentation': {
            'display_name': 'Cell Segmentation', 
            'description': 'Test cell segmentation model',
            'models': {
                'cellpose_v1': {
                    'display_name': 'Cellpose v1',
                    'description': 'Test cellpose model',
                    'variants': {
                        'cyto': {
                            'display_name': 'Cytoplasm',
                            'preprocessing': ['normalize']
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def dummy_settings():
    """Create dummy settings for testing."""
    return {
        'data_selection': {
            'last_directory': str(Path.home()),
            'file_patterns': ['*.tif', '*.tiff', '*.png']
        },
        'model_selection': {
            'selected_task': 'organelle_segmentation',
            'selected_model': 'mitochondria_v1',
            'selected_variant': 'default'
        },
        'export': {
            'export_format': 'tiff',
            'output_directory': str(Path.home() / 'outputs')
        }
    }


@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_labels_data():
    """Create sample labels data for testing."""
    return np.random.randint(0, 10, (100, 100), dtype=np.uint16)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_plugin_manager():
    """Mock the napari plugin manager."""
    with patch('npe2.PluginManager.instance') as mock_pm, \
         patch('napari.plugins._initialize_plugins') as mock_init:
        pm_instance = Mock()
        pm_instance.commands.execute.side_effect = lambda cmd: {
            'ai-on-demand.get_manifests': {},
            'ai-on-demand.get_settings': {}
        }.get(cmd, {})
        # Mock iter_manifests to return an empty iterable
        pm_instance.iter_manifests.return_value = []
        mock_pm.return_value = pm_instance
        # Mock plugin initialization to avoid conflicts
        mock_init.return_value = None
        yield pm_instance


@pytest.fixture
def mock_aiod_utils():
    """Mock the aiod_utils modules to avoid import errors."""
    with patch.dict('sys.modules', {
        'aiod_utils': Mock(),
        'aiod_utils.io': Mock(),
        'aiod_utils.preprocess': Mock(),
        'aiod_utils.rle': Mock(),
        'aiod_registry': Mock(),
        'aiod_registry.registry': Mock()
    }):
        yield


@pytest.fixture
def mock_file_dialog():
    """Mock QFileDialog for headless testing."""
    def _mock_get_open_filenames(parent=None, caption='', directory='', filter=''):
        """Mock getOpenFileNames to return test files."""
        return (['/test/file1.tif', '/test/file2.tif'], 'All Files (*)')
    
    def _mock_get_existing_directory(parent=None, caption='', directory=''):
        """Mock getExistingDirectory to return test directory."""
        return '/test/directory'
    
    def _mock_get_save_filename(parent=None, caption='', directory='', filter=''):
        """Mock getSaveFileName to return test save location."""
        return ('/test/output.tif', 'TIFF Files (*.tif)')
    
    with patch('qtpy.QtWidgets.QFileDialog.getOpenFileNames', side_effect=_mock_get_open_filenames), \
         patch('qtpy.QtWidgets.QFileDialog.getExistingDirectory', side_effect=_mock_get_existing_directory), \
         patch('qtpy.QtWidgets.QFileDialog.getSaveFileName', side_effect=_mock_get_save_filename):
        yield


@pytest.fixture
def mock_nextflow():
    """Mock Nextflow execution for testing inference workflows."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Nextflow execution completed successfully"
        yield mock_run


@pytest.fixture
def populated_napari_viewer(sample_image_data, sample_labels_data):
    """Create a napari viewer with sample data for testing."""
    viewer = napari.Viewer(show=False)
    viewer.add_image(sample_image_data, name='test_image')
    viewer.add_labels(sample_labels_data, name='test_labels')
    yield viewer
    viewer.close()


@pytest.fixture(autouse=True)
def setup_qt_environment(qapp):
    """Ensure Qt application is available for all tests."""
    pass