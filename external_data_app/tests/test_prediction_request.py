import pytest
import pandas as pd
from unittest.mock import patch
from prediction_request import load_csv_file, select_data_range, convert_to_json, send_post_request

def test_load_csv_file():
    df = load_csv_file('data/database_input2.csv')
    assert isinstance(df, pd.DataFrame)

def test_select_data_range():
    df = pd.DataFrame({'A': range(100)})
    selected_df = select_data_range(df, 0.3, 0.7)
    assert len(selected_df) == 40  # 40% of 100

def test_convert_to_json():
    df = pd.DataFrame({'A': [1, 2, 3]})
    json_data = convert_to_json(df)
    assert json_data == '[{"A":1},{"A":2},{"A":3}]'

@patch('prediction_request.requests.post')
def test_send_post_request(mock_post):
    mock_post.return_value.json.return_value = {"prediction": "success"}
    json_data = '[{"A":1},{"A":2},{"A":3}]'
    response = send_post_request(json_data, 'http://localhost:8080/predict/')
    assert response == {"prediction": "success"}
    mock_post.assert_called_once_with('http://localhost:8080/predict/', headers={'Content-Type': 'application/json'}, data=json_data)