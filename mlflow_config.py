"""
MLflow Configuration for Medical Summarizer AI
Manages experiment tracking, model registry, and performance monitoring
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLflowConfig:
    """Configuration and utilities for MLflow integration"""
    
    def __init__(self, experiment_name: str = "medical-summarizer"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            # Set tracking URI (local filesystem by default)
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow tracking.")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run"""
        try:
            return mlflow.start_run(run_name=run_name, tags=tags or {})
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            return None
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to current run"""
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_model(self, model, artifact_path: str, registered_model_name: str = None):
        """Log model to MLflow"""
        try:
            mlflow.log_artifacts(model, artifact_path)
            if registered_model_name:
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}", 
                                   registered_model_name)
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def log_text(self, text: str, artifact_path: str):
        """Log text as artifact"""
        try:
            mlflow.log_text(text, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log text: {e}")
    
    def end_run(self):
        """End current MLflow run"""
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

# Global MLflow configuration
mlflow_config = MLflowConfig() 