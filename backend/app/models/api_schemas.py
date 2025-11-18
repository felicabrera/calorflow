from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel
import numpy as np
import pandas as pd


# ----------------------------- Helper functions ----------------------------

def numpy_to_python(obj):
    """
    Convert numpy arrays/scalars into python native types for JSON serialization.
    """
    # numpy arrays -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar -> python scalar
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    # dict with numpy values
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    return obj


def dataframe_to_features(df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
    """
    Convert DataFrame to (features, feature_names) suitable for API payloads.
    """
    if feature_cols is None:
        exclude_cols = ['PCI', 'H2', 'sampled_date', 'timestamp', 'id', 'ID']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    features = df[feature_cols].values.tolist()
    return features, feature_cols


def features_to_dataframe(features: List[List[float]], feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert features (list of lists) + names into a pandas DataFrame.
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(features[0]))]

    return pd.DataFrame(features, columns=feature_names)


# ---------------------------- Error / Generic models ------------------------

class ErrorDetail(BaseModel):
    error_code: str
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


class ErrorResponse(BaseModel):
    # Pydantic v2 removed Field(const=True); use Literal to enforce constant values
    success: Literal[False] = False
    error: ErrorDetail


# ------------------------------- Health / Meta ----------------------------

class ModelAvailability(BaseModel):
    FCC: bool
    CCR: bool


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: Optional[str]
    models_available: ModelAvailability
    dependencies: Optional[Dict[str, bool]] = None
    message: Optional[str] = None


# ------------------------------- Model list --------------------------------

class ModelInfo(BaseModel):
    process_name: str
    model_path: str
    model_files: Optional[List[str]]
    training_history: Optional[Dict[str, Any]]
    created_at: Optional[str]
    file_size_mb: Optional[float]


class ModelListResponse(BaseModel):
    success: bool = True
    models: List[ModelInfo]
    total_models: int


# ------------------------------- Training ----------------------------------

class TrainingConfigSchema(BaseModel):
    random_seed: Optional[int] = 42
    n_trials: Optional[int] = 300
    cv_folds: Optional[int] = 5
    use_time_series_cv: Optional[bool] = True
    use_autogluon: Optional[bool] = True
    autogluon_time_limit: Optional[int] = 3600
    models_to_train: Optional[List[str]] = None


class TrainingRequest(BaseModel):
    process_name: str
    data_path: Optional[str] = None
    output_dir: Optional[str] = 'models'
    features: Optional[List[List[float]]] = None
    feature_names: Optional[List[str]] = None
    targets_pci: Optional[List[float]] = None
    targets_h2: Optional[List[float]] = None
    config: Optional[TrainingConfigSchema] = None


class TrainingResponse(BaseModel):
    success: bool = True
    process_name: str
    metrics_pci: Dict[str, Any]
    metrics_h2: Dict[str, Any]
    competition_score: Dict[str, Any]
    training_time: float
    n_samples_train: int
    n_samples_test: int
    model_path: str
    message: Optional[str]


# ------------------------------- Prediction --------------------------------

class PredictionRequest(BaseModel):
    process_name: str
    data_path: Optional[str] = None
    features: Optional[List[List[float]]] = None
    feature_names: Optional[List[str]] = None
    model_dir: Optional[str] = 'models'
    return_features: Optional[bool] = False


class PredictionResponse(BaseModel):
    success: bool
    process_name: str
    predictions_pci: List[float]
    predictions_h2: List[float]
    n_samples: int
    statistics: Dict[str, Any]
    features: Optional[Any] = None
    timestamps: Optional[List[str]] = None


class BatchPredictionRequest(BaseModel):
    process_name: str
    file_paths: List[str]
    model_dir: Optional[str] = 'models'
    output_dir: Optional[str] = 'predictions'


class BatchPredictionResponse(BaseModel):
    success: bool
    process_name: str
    total_files: int
    successful: int
    failed: int
    output_files: List[str]
    errors: Optional[Dict[str, str]] = None


# ------------------------------ Submission ---------------------------------

class SubmissionRequest(BaseModel):
    process_name: str
    model_dir: str
    data_dir: str
    output_dir: str


class SubmissionResponse(BaseModel):
    success: bool
    process_name: str
    submission_file: str
    n_samples: int
    pci_range: Tuple[float, float]
    h2_range: Tuple[float, float]
    timestamp: str


# --------------------------- Data Quality ----------------------------------

class DataQualityRequest(BaseModel):
    data_path: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    target_cols: Optional[List[str]] = None


class DataQualityResponse(BaseModel):
    success: bool
    n_samples: int
    n_features: int
    missing_values: Dict[str, int]
    missing_percentage: Any
    duplicates: int
    numeric_features: int
    categorical_features: int
    datetime_features: int
    quality_score: float
    recommendations: List[str]


# Export names
__all__ = [
    'ErrorDetail', 'ErrorResponse',
    'HealthCheckResponse', 'ModelAvailability',
    'ModelInfo', 'ModelListResponse',
    'TrainingRequest', 'TrainingResponse', 'TrainingConfigSchema',
    'PredictionRequest', 'PredictionResponse',
    'BatchPredictionRequest', 'BatchPredictionResponse',
    'SubmissionRequest', 'SubmissionResponse',
    'DataQualityRequest', 'DataQualityResponse',
    'numpy_to_python', 'dataframe_to_features', 'features_to_dataframe'
]
