import os
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, ConfigDict
import argparse
from prediction_model.config import config
from prediction_model.pipeline.predictions import load_model, preprocess_data_from_json, make_prediction

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
model = load_model(os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME))

class PredictionRequest(BaseModel):
    Contract: str
    tenure: int
    TotalCharges: str
    PhoneService: str = Field(None)

    model_config = ConfigDict(extra='allow')

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    errors = exc.errors()
    missing_fields = [error['loc'][0] for error in errors if error['type'] == 'value_error.missing']
    message = f"Request lacks the following required fields: {', '.join(missing_fields)}"
    return JSONResponse(
        status_code=422,
        content={"message": message, "detail": errors},
    )

@app.post("/predict/")
async def predict(request: list[PredictionRequest], background_tasks: BackgroundTasks):
    try:
        json_data = [item.model_dump() for item in request]
        # Run the preprocessing pipeline with the JSON input
        preprocessed_data_file = preprocess_data_from_json(
            json_data=json_data,
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