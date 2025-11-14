"""
API Module for Backend Integration
Provides schemas, endpoints, helpers, and logging for backend frameworks
"""

from .api_schemas import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    SubmissionRequest, SubmissionResponse,
    DataQualityRequest, DataQualityResponse,
    HealthCheckResponse, ModelListResponse,
    ErrorResponse, ErrorDetail
)

from .api_endpoints import (
    train_model_endpoint,
    predict_endpoint,
    batch_predict_endpoint,
    generate_submission_endpoint,
    check_data_quality_endpoint,
    list_models_endpoint,
    health_check_endpoint
)

from .api_helpers import (
    serialize_predictions,
    serialize_training_results,
    create_error_response,
    handle_api_errors,
    validate_process_name,
    validate_model_exists
)

from .logging_config import (
    setup_logging,
    setup_production_logging,
    setup_development_logging,
    get_logger
)

__all__ = [
    # Schemas
    'TrainingRequest', 'TrainingResponse',
    'PredictionRequest', 'PredictionResponse',
    'BatchPredictionRequest', 'BatchPredictionResponse',
    'SubmissionRequest', 'SubmissionResponse',
    'DataQualityRequest', 'DataQualityResponse',
    'HealthCheckResponse', 'ModelListResponse',
    'ErrorResponse', 'ErrorDetail',
    
    # Endpoints
    'train_model_endpoint',
    'predict_endpoint',
    'batch_predict_endpoint',
    'generate_submission_endpoint',
    'check_data_quality_endpoint',
    'list_models_endpoint',
    'health_check_endpoint',
    
    # Helpers
    'serialize_predictions',
    'serialize_training_results',
    'create_error_response',
    'handle_api_errors',
    'validate_process_name',
    'validate_model_exists',
    
    # Logging
    'setup_logging',
    'setup_production_logging',
    'setup_development_logging',
    'get_logger',
]
