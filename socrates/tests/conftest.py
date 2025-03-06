"""Shared fixtures for pytest."""
import pytest
import sys
import os
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests."""
    return [
        {"id": 1, "name": "Item 1", "value": 10},
        {"id": 2, "name": "Item 2", "value": 20},
        {"id": 3, "name": "Item 3", "value": 30},
    ]
