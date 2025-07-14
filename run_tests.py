#!/usr/bin/env python
"""
Test runner script that sets up mocks before importing modules.
"""
import sys
from unittest.mock import Mock

# Mock all external dependencies
external_modules = [
    'aiod_utils',
    'aiod_utils.io', 
    'aiod_utils.preprocess',
    'aiod_utils.rle',
    'aiod_utils.stacks',
    'aiod_registry',
    'aiod_registry.registry',
    'glasbey',
    'dask_image',
    'dask_image.ndmeasure'
]

for module in external_modules:
    sys.modules[module] = Mock()

# Set up additional mock attributes for more realistic mocking
mock_aiod_io = sys.modules['aiod_utils.io']
mock_aiod_io.extract_idxs_from_fname = Mock(return_value=[0, 1, 2])

mock_aiod_rle = sys.modules['aiod_utils.rle'] 
mock_aiod_rle.encode_rle = Mock(return_value="encoded_rle")
mock_aiod_rle.decode_rle = Mock(return_value=Mock())

mock_aiod_preprocess = sys.modules['aiod_utils.preprocess']
mock_aiod_preprocess.normalize = Mock(return_value=Mock())
mock_preprocess_obj = Mock()
mock_preprocess_obj.tooltip = "Test tooltip"
mock_preprocess_obj.help = "Test help"
mock_aiod_preprocess.get_all_preprocess_methods = Mock(return_value={
    'normalize': {
        'description': 'Normalize image',
        'params': {},
        'object': mock_preprocess_obj
    },
    'resize': {
        'description': 'Resize image', 
        'params': {},
        'object': mock_preprocess_obj
    }
})
mock_aiod_preprocess.get_params_str = Mock(return_value="normalized")

mock_aiod_stacks = sys.modules['aiod_utils.stacks']
mock_aiod_stacks.generate_stack_indices = Mock(return_value=[])
mock_aiod_stacks.calc_num_stacks = Mock(return_value=1)
mock_aiod_stacks.Stack = Mock()

# Mock aiod_registry constants
mock_aiod_registry = sys.modules['aiod_registry']
mock_aiod_registry.TASK_NAMES = {
    'organelle_segmentation': 'Organelle Segmentation',
    'cell_segmentation': 'Cell Segmentation',
    'nuclei_segmentation': 'Nuclei Segmentation'
}

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(sys.argv[1:]))