import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from prediction_model.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict():
    # Example input data
    input_data = [
        {"customerID": "8383-SGHJU", "Contract": "One year", "tenure": 33, "PhoneService": "Yes", "TotalCharges": "1952.8"},
        {"customerID": "7607-QKKTJ", "Contract": "Two year", "tenure": 45, "PhoneService": "Yes", "TotalCharges": "4368.85"}
    ]

    response = client.post("/predict/", json=input_data)
    assert response.status_code == 200
    assert "predictions" in response.json()