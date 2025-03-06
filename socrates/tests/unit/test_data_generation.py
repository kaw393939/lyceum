"""Unit tests for data generation functions."""
import pytest
from rag_demo.main import generate_sample_data

def test_generate_sample_data():
    """Test sample data generation."""
    data = generate_sample_data()
    
    # Check that data has the expected structure
    assert len(data) == 20
    assert "date" in data.columns
    assert "value_a" in data.columns
    assert "value_b" in data.columns
    assert "category" in data.columns
    
    # Check data types
    assert data["date"].dtype.kind == 'M'  # datetime
    assert data["value_a"].dtype.kind == 'f'  # float
    assert data["value_b"].dtype.kind == 'f'  # float
    assert data["category"].dtype.kind in ['O', 'U']  # object or unicode
