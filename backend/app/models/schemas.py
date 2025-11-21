"""
API Schemas for FastAPI endpoints
Pydantic models for request/response validation
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np


# ============================================================================
# MODEL INFO
# ============================================================================

class ModelInfo(BaseModel):
    """Information about a trained model"""
    process: str = Field(..., description="Process type (FCC or CCR)")
    target: str = Field(..., description="Target variable (PCI or H2)")
    model_type: str = Field(..., description="Type of model")
    path: str = Field(..., description="Path to model file")
    available: bool = Field(..., description="Whether model is available")


# ============================================================================
# PREDICTION SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for predictions"""
    process: str = Field(..., description="Process type (FCC or CCR)")
    features: List[List[float]] = Field(..., description="Feature matrix")
    feature_names: List[str] = Field(..., description="Feature column names")


class PredictionResponse(BaseModel):
    """Response from prediction"""
    success: bool = Field(..., description="Prediction success status")
    process: str = Field(..., description="Process type")
    predictions_pci: List[float] = Field(..., description="PCI predictions")
    predictions_h2: List[float] = Field(..., description="H2 predictions")
    n_predictions: int = Field(..., description="Number of predictions")


# ============================================================================
# TRAINING SCHEMAS
# ============================================================================

class TrainingConfig(BaseModel):
    """Configuration for model training"""
    random_seed: int = Field(default=42, description="Random seed")
    n_trials: int = Field(default=300, description="Number of Optuna trials")
    cv_folds: int = Field(default=5, description="Number of CV folds")


class TrainingRequest(BaseModel):
    """Request for model training"""
    process: str = Field(..., description="Process type (FCC or CCR)")
    config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")


class TrainingResponse(BaseModel):
    """Response from model training"""
    success: bool = Field(..., description="Training success status")
    message: str = Field(..., description="Status message")
    task_id: str = Field(..., description="Training task ID")
    status: str = Field(..., description="Training status")


# ============================================================================
# METRICS AND VISUALIZATIONS
# ============================================================================

class MetricsResponse(BaseModel):
    """Response with training metrics"""
    success: bool = Field(..., description="Success status")
    process: str = Field(..., description="Process type")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")


class VisualizationDataResponse(BaseModel):
    """Response with visualization data"""
    success: bool = Field(..., description="Success status")
    process: str = Field(..., description="Process type")
    data: Dict[str, Any] = Field(..., description="Visualization data")