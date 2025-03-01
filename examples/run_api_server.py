"""
Run API Server.

This script demonstrates how to run the FastAPI server for model serving.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path to allow importing from nlp_suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp_suite.model_serving.api import run_server, create_demo_model, load_model_from_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the API server for model serving")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file to load")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def setup_logging(log_level):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("api_server.log"),
        ],
    )


def main():
    """Run the API server."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Log startup information
    logging.info(f"Starting API server on {args.host}:{args.port}")
    if args.model_path:
        logging.info(f"Loading model from {args.model_path}")
    else:
        logging.info("Using demo model")
    
    # Run the server
    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main() 