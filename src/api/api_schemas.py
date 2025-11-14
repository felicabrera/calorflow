"""
API Schemas and Validation for Backend Integration
Pydantic models for request/response validation
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np


# ============================================================================
# TRAINING SCHEMAS
# ============================================================================

class TrainingConfig(BaseModel):
    """Configuration for model training"""
    random_seed: int = Field(default=42, ge=1, description="Random seed for reproducibility")
    n_trials: int = Field(default=300, ge=10, le=1000, description="Number of Optuna trials")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of CV folds")
    use_time_series_cv: bool = Field(default=True, description="Use time series CV")
    use_autogluon: bool = Field(default=False, description="Enable AutoGluon")
    autogluon_time_limit: int = Field(default=3600, ge=300, description="AutoGluon time limit (seconds)")
    tolerance_pct: float = Field(default=0.10, ge=0.0, le=1.0, description="Tolerance percentage")
    enable_oof_ensemble: bool = Field(default=True, description="Enable OOF ensemble")
    models_to_train: List[Literal['xgboost', 'lightgbm', 'catboost', 'randomforest']] = Field(
        default=['xgboost', 'lightgbm', 'catboost', 'randomforest'],
        description="Models to train"
    )


class TrainingRequest(BaseModel):
    """Request for model training"""
    process_name: Literal['FCC', 'CCR'] = Field(..., description="Process type")
    data_path: Optional[str] = Field(None, description="Path to training data CSV")
    features: Optional[List[List[float]]] = Field(None, description="Feature matrix as list of lists")
    targets_pci: Optional[List[float]] = Field(None, description="PCI target values")
    targets_h2: Optional[List[float]] = Field(None, description="H2 target values")
    feature_names: Optional[List[str]] = Field(None, description="Feature column names")
    config: Optional[TrainingConfig] = Field(default_factory=TrainingConfig, description="Training configuration")
    output_dir: str = Field(default='models', description="Directory to save models")
    
    @field_validator('features', 'targets_pci', 'targets_h2')
    @classmethod
    def check_not_empty(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("Cannot be empty if provided")
        return v
    
    @field_validator('features')
    @classmethod
    def check_features_shape(cls, v):
        if v is not None:
            if len(v) > 0:
                first_len = len(v[0])
                if not all(len(row) == first_len for row in v):
                    raise ValueError("All feature rows must have same length")
        return v


class TrainingResponse(BaseModel):
    """Response from model training"""
    success: bool = Field(..., description="Training success status")
    process_name: str = Field(..., description="Process name")
    metrics_pci: Dict[str, float] = Field(..., description="PCI metrics")
    metrics_h2: Dict[str, float] = Field(..., description="H2 metrics")
    competition_score: Dict[str, float] = Field(..., description="Competition score")
    training_time: float = Field(..., description="Training time in seconds")
    n_samples_train: int = Field(..., description="Number of training samples")
    n_samples_test: int = Field(..., description="Number of test samples")
    model_path: str = Field(..., description="Path to saved models")
    message: Optional[str] = Field(None, description="Additional message")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# PREDICTION SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for predictions"""
    process_name: Literal['FCC', 'CCR'] = Field(..., description="Process type")
    data_path: Optional[str] = Field(None, description="Path to data CSV for prediction")
    features: Optional[List[List[float]]] = Field(None, description="Feature matrix as list of lists")
    feature_names: Optional[List[str]] = Field(None, description="Feature column names")
    model_dir: str = Field(default='models', description="Directory containing trained models")
    return_features: bool = Field(default=False, description="Include features in response")
    
    @field_validator('features')
    @classmethod
    def check_not_empty(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("Features cannot be empty if provided")
        return v


class PredictionResponse(BaseModel):
    """Response from prediction"""
    success: bool = Field(..., description="Prediction success status")
    process_name: str = Field(..., description="Process name")
    predictions_pci: List[float] = Field(..., description="PCI predictions")
    predictions_h2: List[float] = Field(..., description="H2 predictions")
    n_samples: int = Field(..., description="Number of samples predicted")
    statistics: Dict[str, Any] = Field(..., description="Prediction statistics")
    features: Optional[List[List[float]]] = Field(None, description="Input features (if requested)")
    timestamps: Optional[List[str]] = Field(None, description="Timestamps if available")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    process_name: Literal['FCC', 'CCR'] = Field(..., description="Process type")
    file_paths: List[str] = Field(..., min_length=1, description="List of file paths to predict")
    model_dir: str = Field(default='models', description="Directory containing trained models")
    output_dir: str = Field(default='predictions', description="Directory to save predictions")


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction"""
    success: bool = Field(..., description="Batch prediction success status")
    process_name: str = Field(..., description="Process name")
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
    output_files: List[str] = Field(..., description="List of output file paths")
    errors: Optional[Dict[str, str]] = Field(None, description="Errors by file path")


# ============================================================================
# COMPETITION SUBMISSION SCHEMAS
# ============================================================================

class SubmissionRequest(BaseModel):
    """Request for competition submission generation"""
    process_name: Literal['FCC', 'CCR'] = Field(..., description="Process type")
    model_dir: str = Field(default='models', description="Directory containing trained models")
    data_dir: str = Field(default='data', description="Directory containing test data")
    output_dir: str = Field(default='predictions', description="Directory to save submission")


class SubmissionResponse(BaseModel):
    """Response from submission generation"""
    success: bool = Field(..., description="Submission generation success status")
    process_name: str = Field(..., description="Process name")
    submission_file: str = Field(..., description="Path to submission file")
    n_samples: int = Field(..., description="Number of samples in submission")
    pci_range: tuple[float, float] = Field(..., description="PCI prediction range (min, max)")
    h2_range: tuple[float, float] = Field(..., description="H2 prediction range (min, max)")
    timestamp: str = Field(..., description="Submission timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# MODEL MANAGEMENT SCHEMAS
# ============================================================================

class ModelInfo(BaseModel):
    """Information about a trained model"""
    process_name: str = Field(..., description="Process name")
    model_path: str = Field(..., description="Path to model directory")
    model_files: List[str] = Field(..., description="List of model files")
    training_history: Optional[Dict[str, Any]] = Field(None, description="Training history")
    created_at: Optional[str] = Field(None, description="Model creation timestamp")
    file_size_mb: Optional[float] = Field(None, description="Total size in MB")


class ModelListResponse(BaseModel):
    """Response for listing available models"""
    success: bool = Field(..., description="Success status")
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_models: int = Field(..., description="Total number of models")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: Literal['healthy', 'degraded', 'unhealthy'] = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    models_available: Dict[str, bool] = Field(..., description="Model availability by process")
    dependencies: Dict[str, bool] = Field(..., description="Dependency availability")
    version: str = Field(default='2.0.0', description="API version")
    message: Optional[str] = Field(None, description="Status message")


# ============================================================================
# DATA VALIDATION SCHEMAS
# ============================================================================

class DataQualityRequest(BaseModel):
    """Request for data quality check"""
    data_path: Optional[str] = Field(None, description="Path to data CSV")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Data as list of dicts")
    target_cols: List[str] = Field(default=['PCI', 'H2'], description="Target column names")


class DataQualityResponse(BaseModel):
    """Response from data quality check"""
    success: bool = Field(..., description="Check success status")
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features")
    missing_values: Dict[str, int] = Field(..., description="Missing values per column")
    missing_percentage: Dict[str, float] = Field(..., description="Missing percentage per column")
    duplicates: int = Field(..., description="Number of duplicate rows")
    numeric_features: int = Field(..., description="Number of numeric features")
    categorical_features: int = Field(..., description="Number of categorical features")
    datetime_features: int = Field(..., description="Number of datetime features")
    outliers: Optional[Dict[str, int]] = Field(None, description="Outliers per column")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score (0-100)")
    recommendations: List[str] = Field(..., description="Data quality recommendations")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============================================================================
# ERROR SCHEMAS
# ============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    traceback: Optional[str] = Field(None, description="Stack trace (debug mode only)")


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(default=False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def dataframe_to_features(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> tuple[List[List[float]], List[str]]:
    """
    Convert DataFrame to feature format for API
    
    Args:
        df: Input DataFrame
        feature_cols: Feature column names (if None, use all numeric columns)
    
    Returns:
        Tuple of (features as list of lists, feature names)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    features = df[feature_cols].values.tolist()
    return features, feature_cols


def features_to_dataframe(features: List[List[float]], feature_names: List[str]) -> pd.DataFrame:
    """
    Convert feature format from API to DataFrame
    
    Args:
        features: Features as list of lists
        feature_names: Feature column names
    
    Returns:
        DataFrame with features
    """
    return pd.DataFrame(features, columns=feature_names)


def numpy_to_python(obj: Any) -> Any:
    """
    Convert numpy types to Python types for JSON serialization
    
    Args:
        obj: Object to convert
    
    Returns:
        Python-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


if __name__ == '__main__':
    print("API Schemas for Backend Integration")
    print("\nAvailable schemas:")
    print("  - TrainingRequest/Response")
    print("  - PredictionRequest/Response")
    print("  - BatchPredictionRequest/Response")
    print("  - SubmissionRequest/Response")
    print("  - ModelInfo/ModelListResponse")
    print("  - HealthCheckResponse")
    print("  - DataQualityRequest/Response")
    print("  - ErrorDetail/ErrorResponse")
