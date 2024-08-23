import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
from prediction_model.pipeline.preprocessing import run_pipeline_from_json  # Adjusted to use the JSON version of the pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prediction_model.config import config


def load_model(model_path):
    """Load the trained model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Loaded model is not a RandomForestClassifier.")
    return model

def preprocess_data_from_json(json_data, output_file, db_suffix):
    """Run the preprocessing pipeline with JSON input and return the path to the preprocessed data."""
    pipeline_success = run_pipeline_from_json(json_data, output_file, db_suffix)
    if not pipeline_success:
        raise RuntimeError("Pipeline execution failed.")
    return output_file + db_suffix

def make_prediction(model, preprocessed_data_file):
    """Run predictions using the pre-trained model on preprocessed data."""
    data = pd.read_csv(preprocessed_data_file)  # Adjust if you want to directly process JSON in-memory
    predictions = model.predict(data)
    return predictions

def save_predictions(predictions, output_file, suffix=".csv"):
    """Save the predictions to a file."""
    predictions_file = f"{output_file}_predictions{suffix}"
    predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
    predictions_df.to_csv(predictions_file, index=False)

    print(f"Predictions saved to {predictions_file}")

def main():
    # Assume we have JSON data instead of a CSV file
    json_data = [
        {"Unnamed: 0": 2112, "customerID": "8383-SGHJU", "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 33, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No", "PaymentMethod": "Electronic check", "MonthlyCharges": 59.4, "TotalCharges": "1952.8"},
        {"Unnamed: 0": 2113, "customerID": "7607-QKKTJ", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 45, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "Yes", "Contract": "One year", "PaperlessBilling": "Yes", "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 95.0, "TotalCharges": "4368.85"}
    ]

    output_file = os.path.join(config.DATA_PATH, 'outputs/processed_output')
    preprocessed_data_file = preprocess_data_from_json(json_data, output_file, config.DB_SUFFIX)

    model_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model = load_model(model_path)

    predictions = make_prediction(model, preprocessed_data_file)
    save_predictions(predictions, output_file)
    print(predictions)

if __name__ == '__main__':
    main()