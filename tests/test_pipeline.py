import os
import pytest
from fastapi.testclient import TestClient
from src.app import app, MODEL_PATH
import pandas as pd

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Customer Churn" in data["service"]

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_data_generation_and_loading():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "churn_data.csv")
    if not os.path.exists(data_path):
        from data.generate_data import generate_churn_dataset
        generate_churn_dataset(output_path=data_path)
    
    df = pd.read_csv(data_path)
    assert not df.empty
    assert "churn" in df.columns
    assert len(df) >= 100

def test_prediction_endpoint():
    # If model is not trained yet, train it for the test
    if not os.path.exists(MODEL_PATH):
        from src.train import train
        train()
        
    payload = {
        "tenure": 2,
        "monthly_charges": 89.5,
        "total_charges": 179.0,
        "contract_type": 0,
        "tech_support": 0,
        "online_security": 0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0

def test_prediction_invalid_input():
    # Invalid tenure (negative value)
    payload = {
        "tenure": -5,
        "monthly_charges": 89.5,
        "total_charges": 179.0,
        "contract_type": 0,
        "tech_support": 0,
        "online_security": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Validation error
