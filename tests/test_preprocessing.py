"""
Tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from preprocessing import (
    optimize_dtypes,
    handle_missing_values,
    drop_cols_with_many_nans,
    encode_categorical
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'numeric_col': [1.0, 2.0, np.nan, 4.0],
        'categorical_col': ['A', 'B', np.nan, 'C'],
        'int_col': [1, 2, 3, 4],
        'many_nans': [1, np.nan, np.nan, np.nan]
    })

def test_optimize_dtypes(sample_data):
    """Test dtype optimization."""
    optimized = optimize_dtypes(sample_data)
    assert optimized['numeric_col'].dtype == 'float32'
    assert optimized['int_col'].dtype == 'int32'

def test_handle_missing_values(sample_data):
    """Test missing value handling."""
    processed = handle_missing_values(sample_data)
    assert not processed['numeric_col'].isna().any()
    assert not processed['categorical_col'].isna().any()

def test_drop_cols_with_many_nans(sample_data):
    """Test dropping columns with too many NaNs."""
    cleaned, dropped_cols = drop_cols_with_many_nans(sample_data, threshold=0.7)
    assert 'many_nans' in dropped_cols
    assert 'many_nans' not in cleaned.columns

def test_encode_categorical(sample_data):
    """Test categorical encoding."""
    encoded, encoders = encode_categorical(sample_data, fit=True)
    assert 'categorical_col' in encoders
    assert encoded['categorical_col'].dtype in ['int32', 'int64']
