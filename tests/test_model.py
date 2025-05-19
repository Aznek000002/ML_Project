"""
Tests for the model training and prediction module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from train_model import train_model, evaluate_model, train_single_model
from predict import Predictor
import os
from config import MODEL_FREQ_PATH, MODEL_CM_PATH

@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'categorical': ['A', 'B'] * 50
    })
    y = pd.Series(np.random.rand(100), name='target')
    return X, y

def test_train_model(sample_training_data):
    """Test model training function."""
    X, y = sample_training_data
    model = train_model(X, pd.DataFrame({'target': y}), 'target')
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, 'predict')

def test_evaluate_model(sample_training_data):
    """Test model evaluation function."""
    X, y = sample_training_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    metrics = evaluate_model(model, X, y)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    
    # Check that metrics are valid
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['r2'] <= 1

def test_train_single_model(sample_training_data, tmp_path):
    """Test single model training and saving."""
    X, y = sample_training_data
    model_path = tmp_path / "test_model.pkl"
    
    train_single_model(X, y, str(model_path), 'test')
    
    # Check if model file was created
    assert model_path.exists()
    
    # Try loading the model
    import pickle
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert isinstance(loaded_model, RandomForestRegressor)
