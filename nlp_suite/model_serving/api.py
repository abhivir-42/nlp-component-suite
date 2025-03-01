"""
Model Serving API module.

This module provides a FastAPI-based API for serving models in production
with proper monitoring, validation, and error handling.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """
    Model for prediction request data.
    
    This defines the schema for the input data received by the API.
    The schema should be adapted to match the specific model requirements.
    """

    inputs: List[List[float]] = Field(
        ...,
        description="List of input feature vectors. Each vector should have the same length.",
        example=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of the model to use for prediction. If not provided, the default model will be used.",
        example="mlp_default",
    )


class PredictionResponse(BaseModel):
    """
    Model for prediction response data.
    
    This defines the schema for the output data returned by the API.
    The schema should be adapted to match the specific model outputs.
    """

    predictions: List[List[float]] = Field(
        ...,
        description="List of prediction vectors. Each vector corresponds to one input vector.",
        example=[[0.8, 0.2], [0.3, 0.7]],
    )
    model_name: str = Field(
        ...,
        description="Name of the model that was used for prediction.",
        example="mlp_default",
    )
    prediction_time_ms: float = Field(
        ...,
        description="Time taken to generate the predictions in milliseconds.",
        example=42.5,
    )


class ModelMetadata(BaseModel):
    """
    Model for model metadata.
    
    This defines the schema for metadata about the available models.
    """

    name: str = Field(
        ...,
        description="Name of the model.",
        example="mlp_default",
    )
    version: str = Field(
        ...,
        description="Version of the model.",
        example="1.0.0",
    )
    description: str = Field(
        ...,
        description="Description of the model.",
        example="Default MLP model for binary classification.",
    )
    input_shape: List[int] = Field(
        ...,
        description="Expected shape of the input data.",
        example=[3],
    )
    output_shape: List[int] = Field(
        ...,
        description="Shape of the output data.",
        example=[2],
    )
    created_at: str = Field(
        ...,
        description="Date and time when the model was created.",
        example="2023-04-01T12:00:00",
    )


class ModelRegistry:
    """
    Model registry for managing multiple models.
    
    This class provides a simple registry for loading and accessing models.
    In a production environment, this would typically interface with a model
    store like MLflow, BentoML, or custom cloud storage.
    """

    def __init__(self) -> None:
        """Initialize the model registry."""
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}

    def register_model(
        self, name: str, model: Any, metadata: Optional[ModelMetadata] = None
    ) -> None:
        """
        Register a model with the registry.
        
        Args:
            name: Name of the model
            model: The model object
            metadata: Optional metadata about the model
        """
        self.models[name] = model
        
        if metadata is None:
            # Create default metadata
            metadata = ModelMetadata(
                name=name,
                version="1.0.0",
                description=f"Model {name}",
                input_shape=[0],  # Placeholder
                output_shape=[0],  # Placeholder
                created_at=datetime.now().isoformat(),
            )
        
        self.metadata[name] = metadata
        logger.info(f"Registered model: {name}")

    def get_model(self, name: Optional[str] = None) -> Tuple[str, Any]:
        """
        Get a model from the registry.
        
        Args:
            name: Name of the model to get. If None, returns the first available model.
            
        Returns:
            Tuple of (model_name, model_object)
            
        Raises:
            HTTPException: If the model is not found or no models are available
        """
        if not self.models:
            raise HTTPException(status_code=503, detail="No models available")
        
        if name is None:
            # Return the first model if no name is specified
            name = next(iter(self.models.keys()))
        
        if name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        
        return name, self.models[name]

    def get_metadata(self, name: Optional[str] = None) -> List[ModelMetadata]:
        """
        Get metadata for models in the registry.
        
        Args:
            name: Name of the model to get metadata for. If None, returns metadata for all models.
            
        Returns:
            List of model metadata
            
        Raises:
            HTTPException: If the model is not found or no models are available
        """
        if not self.metadata:
            raise HTTPException(status_code=503, detail="No models available")
        
        if name is None:
            # Return metadata for all models if no name is specified
            return list(self.metadata.values())
        
        if name not in self.metadata:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        
        return [self.metadata[name]]


# Create the FastAPI app
app = FastAPI(
    title="NLP Component Suite API",
    description="API for serving language models from the NLP Component Suite",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create model registry
model_registry = ModelRegistry()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to add processing time header to responses.
    
    This is useful for monitoring the API performance.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint for health check.
    
    Returns:
        Dict with status message
    """
    return {"status": "ok", "message": "NLP Component Suite API is running"}


@app.get("/models", tags=["Models"], response_model=List[ModelMetadata])
async def list_models():
    """
    List available models.
    
    Returns:
        List of model metadata
    """
    return model_registry.get_metadata()


@app.get("/models/{model_name}", tags=["Models"], response_model=ModelMetadata)
async def get_model_metadata(model_name: str):
    """
    Get metadata for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model metadata
    """
    return model_registry.get_metadata(model_name)[0]


@app.post("/predict", tags=["Predictions"], response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using a model.
    
    Args:
        request: Prediction request with input data
        
    Returns:
        Prediction response with model outputs
    """
    try:
        # Get the model
        model_name, model = model_registry.get_model(request.model_name)
        
        # Log the prediction request
        logger.info(f"Prediction request for model: {model_name}")
        
        # Convert inputs to expected format
        inputs = request.inputs
        
        # Measure prediction time
        start_time = time.time()
        
        # Make predictions
        predictions = []
        for input_vector in inputs:
            # Call the model
            # Note: In a real implementation, this would use the actual model predict method
            # This is a placeholder for demonstration
            output = model(input_vector)
            
            # Convert output to list if it's not already
            if not isinstance(output, list):
                output = [output]
            
            # Extract data from Value objects if necessary
            prediction = [
                o.data if hasattr(o, "data") else float(o) for o in output
            ]
            predictions.append(prediction)
        
        # Calculate prediction time in milliseconds
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Return the prediction response
        return PredictionResponse(
            predictions=predictions,
            model_name=model_name,
            prediction_time_ms=prediction_time_ms,
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log the error
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        
        # Return an error response
        raise HTTPException(
            status_code=500,
            detail=f"Error processing prediction request: {str(e)}",
        )


def load_model_from_path(model_path: str, model_name: Optional[str] = None) -> str:
    """
    Load a model from the given path and register it with the registry.
    
    Args:
        model_path: Path to the model file
        model_name: Optional name for the model. If not provided, will use the filename.
        
    Returns:
        Name of the registered model
    """
    try:
        # In a real implementation, this would load the actual model
        # This is a placeholder for demonstration
        
        # Extract model name from path if not provided
        if model_name is None:
            model_name = os.path.basename(model_path).split(".")[0]
        
        # Create a dummy model for demonstration
        # In a real implementation, this would load the model from the path
        from nlp_suite.architectures.mlp import MLP
        
        # Create a dummy model for demonstration
        model = MLP(nin=3, nouts=[4, 2])
        
        # Create metadata
        metadata = ModelMetadata(
            name=model_name,
            version="1.0.0",
            description=f"Model loaded from {model_path}",
            input_shape=[3],  # Example
            output_shape=[2],  # Example
            created_at=datetime.now().isoformat(),
        )
        
        # Register the model
        model_registry.register_model(model_name, model, metadata)
        
        return model_name
    
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
        raise


def create_demo_model():
    """Create a demo model for testing purposes."""
    from nlp_suite.architectures.mlp import MLP
    
    # Create a simple MLP model
    model = MLP(nin=3, nouts=[4, 2])
    
    # Create metadata
    metadata = ModelMetadata(
        name="demo_model",
        version="1.0.0",
        description="Demo model for testing",
        input_shape=[3],
        output_shape=[2],
        created_at=datetime.now().isoformat(),
    )
    
    # Register the model
    model_registry.register_model("demo_model", model, metadata)


def run_server(host: str = "0.0.0.0", port: int = 8000, model_path: Optional[str] = None):
    """
    Run the API server.
    
    Args:
        host: Host to run the server on
        port: Port to run the server on
        model_path: Optional path to a model file to load
    """
    import uvicorn
    
    # Load model if path is provided
    if model_path:
        load_model_from_path(model_path)
    else:
        # Create a demo model for testing
        create_demo_model()
    
    # Run the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Run the server with a demo model
    run_server() 