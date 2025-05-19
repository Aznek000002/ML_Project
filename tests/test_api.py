"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from main import app

client = TestClient(app)

@pytest.fixture
def sample_request_data():
    """Create sample request data for testing."""
    return {
        "data": [
            {
                "ACTIVIT2": "ACT1",
                "VOCATION": "VOC6",
                "numeric_col": 1.0,
                "categorical_col": "A"
            }
        ]
    }

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]

def test_predict_freq(sample_request_data):
    """Test the frequency prediction endpoint."""
    response = client.post("/predict_freq", json=sample_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)

def test_predict_montant(sample_request_data):
    """Test the montant prediction endpoint."""
    response = client.post("/predict_montant", json=sample_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)

def test_predict_global(sample_request_data):
    """Test the global prediction endpoint."""
    response = client.post("/predict_global", json=sample_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)

def test_invalid_request():
    """Test handling of invalid request data."""
    invalid_data = {"data": []}  # Empty data
    response = client.post("/predict_freq", json=invalid_data)
    assert response.status_code == 500  # Should return error for empty data 