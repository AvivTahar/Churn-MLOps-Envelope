import pytest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prediction_model.pipeline.preprocessing import is_json_empty, run_pipeline_from_json

def test_is_json_empty():
    assert is_json_empty([]) == True
    assert is_json_empty([{}]) == True
    assert is_json_empty([{"key": "value"}]) == False

def test_run_pipeline_from_json():
    json_data = [
        {"customerID": "8383-SGHJU", "Contract": "One year", "tenure": 33, "PhoneService": "Yes", "TotalCharges": "1952.8"},
        {"customerID": "7607-QKKTJ", "Contract": "Two year", "tenure": 45, "PhoneService": "Yes", "TotalCharges": "4368.85"}
    ]
    output_file = "/tmp/test_output"
    db_suffix = ".csv"

    result = run_pipeline_from_json(json_data, output_file, db_suffix)
    assert result == True