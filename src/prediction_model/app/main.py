import os
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import argparse
from prediction_model.config import config
from prediction_model.pipeline.predictions import load_model, preprocess_data, make_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI application to serve predictions.")
    parser.add_argument('--host', type=str, default=app_config['server']['host'], help='Host to run the FastAPI app on.')
    parser.add_argument('--port', type=int, default=app_config['server']['port'], help='Port to run the FastAPI app on.')
    parser.add_argument('--reload', action='store_true', default=app_config['server']['reload'], help='Enable/disable auto-reload.')
    return parser.parse_args()


def load_config(config_file=os.path.join(config.FAST_API_PATH, 'config.yml')):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

app_config = load_config()

app = FastAPI()

# Load the model once at startup
model = load_model(os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME))

class PredictionRequest(BaseModel):
    input_file: str

@app.post("/predict/")
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        # Run the preprocessing pipeline
        preprocessed_data_file = preprocess_data(
            input_file=request.input_file,
            output_file=os.path.join(app_config['model']['output_dir'], 'processed_output'),
            db_suffix='.csv'
        )

        # Make predictions
        predictions = make_prediction(model, preprocessed_data_file)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
