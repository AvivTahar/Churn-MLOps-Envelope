import os
import pandas as pd
import json
import requests
from argparse import ArgumentParser

def load_csv_file(file_path):
    """Load the CSV file into a pandas DataFrame."""
    return pd.read_csv(os.path.join(os.getcwd(), file_path))

def select_data_range(df, from_percentage, to_percentage):
    """Select a portion of the DataFrame based on the given percentage range."""
    total_rows = len(df)
    start_row = int(from_percentage * total_rows)
    end_row = int(to_percentage * total_rows)
    return df.iloc[start_row:end_row]

def convert_to_json(df):
    """Convert the DataFrame to a JSON object."""
    return df.to_json(orient='records')

def send_post_request(json_data, url):
    """Send a POST request to the FastAPI app with the JSON data."""
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json_data)
    return response.json()

def main(file_name, from_percentage, to_percentage, url):

    df = load_csv_file(file_name)
    selected_df = select_data_range(df, from_percentage, to_percentage)
    json_data = convert_to_json(selected_df)
    response = send_post_request(json_data, url)
    
    # Print the response
    print("Response from the server:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    parser = ArgumentParser(description="Send a CSV file data for predictions via a POST request.")
    parser.add_argument('--file_name', type=str, default='external_data_app/database_input2.csv', help="The path to the CSV file.")
    parser.add_argument('--from_percentage', type=float, default=0.3, help="The starting percentage of data to use.")
    parser.add_argument('--to_percentage', type=float, default=0.7, help="The ending percentage of data to use.")
    parser.add_argument('--url', type=str, default="http://localhost:8080/predict/", help="The URL of the FastAPI prediction endpoint.")

    args = parser.parse_args()
    
    main(args.file_name, args.from_percentage, args.to_percentage, args.url)