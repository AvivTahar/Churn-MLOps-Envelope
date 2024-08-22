import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
from prediction_model.pipeline.preprocessing import run_pipeline  # Adjust the import if preprocessing.py is in a different location

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prediction_model.config import config


def load_model(model_path):
    """Load the trained model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Loaded model is not a RandomForestClassifier.")
    return model

def preprocess_data(input_file, output_file, db_suffix):
    """Run the preprocessing pipeline and return the path to the preprocessed data."""
    pipeline_success = run_pipeline(input_file, output_file, db_suffix)
    if not pipeline_success:
        raise RuntimeError("Pipeline execution failed.")
    return output_file + db_suffix

def make_prediction(model, preprocessed_data_file):
    """Run predictions using the pre-trained model on preprocessed data."""
    data = pd.read_csv(preprocessed_data_file)
    predictions = model.predict(data)
    return predictions

def main():
    # Paths to the input and output files
    input_file = os.path.join(config.DATA_PATH, config.TEST_FILE_ONE)
    output_file = os.path.join(config.DATA_PATH, 'outputs/processed_output')

    # Preprocess the data
    preprocessed_data_file = preprocess_data(input_file, output_file, config.DB_SUFFIX)

    # Load the model
    model_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model = load_model(model_path)

    # Make predictions
    predictions = make_prediction(model, preprocessed_data_file)
    print(predictions)

if __name__ == '__main__':
    main()