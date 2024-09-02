import pandas as pd
import json
import requests
import logging
from argparse import ArgumentParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_file(file_path):
    """Load the CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded CSV file: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file: {file_path}. Error: {e}")
        raise

def select_data_range(df, from_percentage, to_percentage):
    """Select a portion of the DataFrame based on the given percentage range."""
    total_rows = len(df)
    start_row = int(from_percentage * total_rows)
    end_row = int(to_percentage * total_rows)
    logger.info(f"Selecting rows from {start_row} to {end_row} out of {total_rows}")
    return df.iloc[start_row:end_row]

def convert_to_json(df):
    """Convert the DataFrame to a JSON object."""
    json_data = df.to_json(orient='records')
    logger.info(f"Converted DataFrame to JSON")
    return json_data

def send_post_request(json_data, url):
    """Send a POST request to the FastAPI app with the JSON data."""
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json_data)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        logger.info(f"Successfully sent POST request to {url}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send POST request to {url}. Error: {e}")
        raise


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
    parser.add_argument('--file_name', type=str, default='external_data_app/data/database_input2.csv', help="The path to the CSV file.")
    parser.add_argument('--from_percentage', type=float, default=0.3, help="The starting percentage of data to use.")
    parser.add_argument('--to_percentage', type=float, default=0.7, help="The ending percentage of data to use.")
    parser.add_argument('--url', type=str, default="http://localhost:8080/predict/", help="The URL of the FastAPI prediction endpoint.")

    args = parser.parse_args()
    
    main(args.file_name, args.from_percentage, args.to_percentage, args.url)

