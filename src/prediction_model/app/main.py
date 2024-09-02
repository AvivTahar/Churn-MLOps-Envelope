import os
import yaml
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, ConfigDict, model_validator
import argparse
from typing import Optional, Union
from prediction_model.config import config
from prediction_model.pipeline.predictions import load_model, preprocess_data_from_json, make_prediction
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import uvicorn

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

instrumentator = Instrumentator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Expose the Prometheus metrics
    instrumentator.expose(app)
    yield

app = FastAPI(lifespan=lifespan)
instrumentator.instrument(app)

def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI application to serve predictions.")
    parser.add_argument('--host', type=str, default=app_config['server']['host'], help='Host to run the FastAPI app on.')
    parser.add_argument('--port', type=int, default=app_config['server']['port'], help='Port to run the FastAPI app on.')
    parser.add_argument('--reload', action='store_true', default=app_config['server']['reload'], help='Enable/disable auto-reload.')
    return parser.parse_args()

def load_config(config_file=os.path.join(config.FAST_API_PATH, 'config.yml')):
    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Configuration loaded successfully")
    return config

def load_model_with_logging(model_path):
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    return model

app_config = load_config()
model = load_model_with_logging(os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME))

from pydantic import BaseModel, Field, root_validator
from typing import Optional, Union

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Optional, Union

class PredictionRequest(BaseModel):
    Contract: Optional[Union[str, float, int]] = Field(None, description="The type of contract the customer has")
    tenure: Optional[Union[str, float, int]] = Field(None, description="The number of months the customer has been with the company")
    TotalCharges: Optional[Union[str, float, int]] = Field(None, description="Total charges to the customer")
    PhoneService: Optional[Union[str, float, int]] = Field(None, description="Indicates if the customer has phone service")

    model_config = ConfigDict(extra='allow')

    @model_validator(mode='before')
    def check_empty_data(cls, values):
        if not values:
            raise ValueError("Empty data: The request contains no data.")
        return values

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc.errors()}")
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
        logger.info(f"Received prediction request with data: {request}")
        json_data = [item.model_dump() for item in request]
        logger.info(f"Converted request data to JSON: {json_data}")

        # Run the preprocessing pipeline with the JSON input
        preprocessed_data_file = preprocess_data_from_json(
            json_data=json_data,
            output_file=os.path.join(app_config['model']['output_dir'], 'processed_output'),
            db_suffix='.csv'
        )
        logger.info(f"Data preprocessed and saved to: {preprocessed_data_file}")

        # Make predictions
        predictions = make_prediction(model, preprocessed_data_file)
        logger.info(f"Predictions made: {predictions}")

        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    args = parse_args()
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )