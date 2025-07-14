"""
Pytest plugin to mock external dependencies before imports.
"""
import sys
from unittest.mock import Mock

def pytest_configure(config):
    """Configure pytest with necessary mocks."""
    # Mock external dependencies that might not be available
    sys.modules['aiod_utils'] = Mock()
    sys.modules['aiod_utils.io'] = Mock()
    sys.modules['aiod_utils.preprocess'] = Mock()
    sys.modules['aiod_utils.rle'] = Mock()
    sys.modules['aiod_registry'] = Mock()
    sys.modules['aiod_registry.registry'] = Mock()