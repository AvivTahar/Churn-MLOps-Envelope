import os
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pipeline.predictions import load_model, preprocess_data, make_prediction

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

app = FastAPI()

# Load the model once at startup
model = load_model(config['model']['model_path'])

class PredictionRequest(BaseModel):
    input_file: str

@app.post("/predict/")
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        # Run the preprocessing pipeline
        preprocessed_data_file = preprocess_data(
            input_file=request.input_file,
            output_file=os.path.join(config['model']['output_dir'], 'processed_output'),
            db_out_suffix='.csv'
        )

        # Make predictions
        predictions = make_prediction(model, preprocessed_data_file)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config['server']['host'],
        port=config['server']['port'],
        reload=config['server']['reload']
    )