"""
API Endpoint Implementations for Backend Integration
Ready-to-use functions for FastAPI/Flask endpoints
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ..trainer import MLTrainer, TrainerConfig, train_pipeline
from ..predictor import (
    load_trained_models,
    predict_from_features,
    predict_from_raw_data,
    generate_competition_submission,
    predict_multiple_files
)
from ..data_utils import load_data, preprocess_data, check_data_quality
from ..features import create_features, clean_feature_names
from .api_schemas import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    SubmissionRequest, SubmissionResponse,
    DataQualityRequest, DataQualityResponse,
    HealthCheckResponse, ModelListResponse
)
from .api_helpers import (
    handle_api_errors,
    serialize_predictions,
    serialize_training_results,
    serialize_dataframe_predictions,
    validate_process_name,
    validate_model_exists,
    request_to_dataframe,
    csv_to_api_format,
    list_available_models,
    check_dependencies
)


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@handle_api_errors(error_code='TRAINING_ERROR', include_traceback=True)
def train_model_endpoint(request: TrainingRequest) -> Dict[str, Any]:
    """
    Train model endpoint
    
    Args:
        request: TrainingRequest with data and configuration
    
    Returns:
        TrainingResponse dictionary
    """
    # Validate process name
    process_name = validate_process_name(request.process_name)
    
    # Load or prepare data
    if request.data_path:
        # Load from CSV
        df = load_data(request.data_path)
        df = preprocess_data(df, handle_missing='interpolate', remove_outliers=True)
        df = create_features(df)
        df = clean_feature_names(df)
        
        # Separate features and targets
        X = df.drop(['PCI', 'H2'], axis=1, errors='ignore')
        y_pci = df['PCI']
        y_h2 = df['H2']
    else:
        # Use provided features/targets
        if not all([request.features, request.targets_pci, request.targets_h2]):
            raise ValueError("Must provide either data_path or (features, targets_pci, targets_h2)")
        
        df = request_to_dataframe(
            features=request.features,
            feature_names=request.feature_names,
            targets_pci=request.targets_pci,
            targets_h2=request.targets_h2
        )
        
        X = df.drop(['PCI', 'H2'], axis=1)
        y_pci = df['PCI']
        y_h2 = df['H2']
    
    # Configure training
    config = TrainerConfig()
    if request.config:
        config.random_seed = request.config.random_seed
        config.n_trials = request.config.n_trials
        config.cv_folds = request.config.cv_folds
        config.use_time_series_cv = request.config.use_time_series_cv
        config.use_autogluon = request.config.use_autogluon
        config.autogluon_time_limit = request.config.autogluon_time_limit
        config.models_to_train = request.config.models_to_train
    
    # Train models
    trainer = train_pipeline(
        X=X,
        y_pci=y_pci,
        y_h2=y_h2,
        process_name=process_name,
        output_dir=request.output_dir,
        config=config
    )
    
    # Serialize results
    return serialize_training_results(
        results=trainer.training_history,
        output_dir=request.output_dir
    )


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@handle_api_errors(error_code='PREDICTION_ERROR', include_traceback=True)
def predict_endpoint(request: PredictionRequest) -> Dict[str, Any]:
    """
    Single prediction endpoint
    
    Args:
        request: PredictionRequest with features or data path
    
    Returns:
        PredictionResponse dictionary
    """
    # Validate
    process_name = validate_process_name(request.process_name)
    validate_model_exists(request.model_dir, process_name)
    
    # Load or prepare data
    if request.data_path:
        # Complete pipeline from CSV
        df_result = predict_from_raw_data(
            data_path=request.data_path,
            model_dir=request.model_dir,
            process_name=process_name
        )
        
        return serialize_dataframe_predictions(df_result, process_name)
    else:
        # Use provided features
        if not request.features:
            raise ValueError("Must provide either data_path or features")
        
        df = request_to_dataframe(
            features=request.features,
            feature_names=request.feature_names
        )
        
        # Predict
        y_pred_pci, y_pred_h2 = predict_from_features(
            X=df,
            model_dir=request.model_dir,
            process_name=process_name
        )
        
        return serialize_predictions(
            predictions_pci=y_pred_pci,
            predictions_h2=y_pred_h2,
            process_name=process_name,
            features=df if request.return_features else None
        )


@handle_api_errors(error_code='BATCH_PREDICTION_ERROR', include_traceback=True)
def batch_predict_endpoint(request: BatchPredictionRequest) -> Dict[str, Any]:
    """
    Batch prediction endpoint
    
    Args:
        request: BatchPredictionRequest with file paths
    
    Returns:
        BatchPredictionResponse dictionary
    """
    # Validate
    process_name = validate_process_name(request.process_name)
    validate_model_exists(request.model_dir, process_name)
    
    # Process files
    errors = {}
    output_files = []
    
    for file_path in request.file_paths:
        try:
            result = predict_from_raw_data(
                data_path=file_path,
                model_dir=request.model_dir,
                process_name=process_name
            )
            
            # Save predictions
            output_path = Path(request.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            input_name = Path(file_path).stem
            output_file = output_path / f'{input_name}_predictions.csv'
            result.to_csv(output_file, index=False)
            
            output_files.append(str(output_file))
        except Exception as e:
            errors[file_path] = str(e)
    
    return {
        'success': True,
        'process_name': process_name,
        'total_files': len(request.file_paths),
        'successful': len(output_files),
        'failed': len(errors),
        'output_files': output_files,
        'errors': errors if errors else None
    }


# ============================================================================
# COMPETITION SUBMISSION ENDPOINT
# ============================================================================

@handle_api_errors(error_code='SUBMISSION_ERROR', include_traceback=True)
def generate_submission_endpoint(request: SubmissionRequest) -> Dict[str, Any]:
    """
    Generate competition submission endpoint
    
    Args:
        request: SubmissionRequest
    
    Returns:
        SubmissionResponse dictionary
    """
    # Validate
    process_name = validate_process_name(request.process_name)
    validate_model_exists(request.model_dir, process_name)
    
    # Generate submission
    submission_file = generate_competition_submission(
        process=process_name,
        model_dir=request.model_dir,
        data_dir=request.data_dir,
        output_dir=request.output_dir
    )
    
    # Read submission to get stats
    import pandas as pd
    submission = pd.read_csv(submission_file)
    
    return {
        'success': True,
        'process_name': process_name,
        'submission_file': submission_file,
        'n_samples': len(submission),
        'pci_range': (float(submission['PCI'].min()), float(submission['PCI'].max())),
        'h2_range': (float(submission['H2'].min()), float(submission['H2'].max())),
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# DATA QUALITY ENDPOINT
# ============================================================================

@handle_api_errors(error_code='DATA_QUALITY_ERROR', include_traceback=True)
def check_data_quality_endpoint(request: DataQualityRequest) -> Dict[str, Any]:
    """
    Data quality check endpoint
    
    Args:
        request: DataQualityRequest
    
    Returns:
        DataQualityResponse dictionary
    """
    # Load data
    if request.data_path:
        df = load_data(request.data_path)
    elif request.data:
        import pandas as pd
        df = pd.DataFrame(request.data)
    else:
        raise ValueError("Must provide either data_path or data")
    
    # Check quality
    quality_report = check_data_quality(df, target_cols=request.target_cols)
    
    # Calculate quality score
    missing_score = 100 - (quality_report['total_missing_pct'])
    duplicate_score = 100 - (quality_report['duplicates'] / len(df) * 100)
    quality_score = (missing_score + duplicate_score) / 2
    
    # Generate recommendations
    recommendations = []
    if quality_report['total_missing_pct'] > 5:
        recommendations.append("High percentage of missing values - consider imputation")
    if quality_report['duplicates'] > 0:
        recommendations.append(f"Found {quality_report['duplicates']} duplicate rows - consider removal")
    if quality_report['numeric_features'] == 0:
        recommendations.append("No numeric features found - check data types")
    if quality_score > 90:
        recommendations.append("Data quality is excellent")
    
    return {
        'success': True,
        'n_samples': quality_report['n_samples'],
        'n_features': quality_report['n_features'],
        'missing_values': quality_report['missing_values'],
        'missing_percentage': {
            col: (count / len(df) * 100) 
            for col, count in quality_report['missing_values'].items()
        },
        'duplicates': quality_report['duplicates'],
        'numeric_features': quality_report['numeric_features'],
        'categorical_features': quality_report['categorical_features'],
        'datetime_features': quality_report['datetime_features'],
        'quality_score': round(quality_score, 2),
        'recommendations': recommendations
    }


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@handle_api_errors(error_code='MODEL_LIST_ERROR', include_traceback=False)
def list_models_endpoint(model_dir: str = 'models') -> Dict[str, Any]:
    """
    List available models endpoint
    
    Args:
        model_dir: Model directory
    
    Returns:
        ModelListResponse dictionary
    """
    models = list_available_models(model_dir)
    
    return {
        'success': True,
        'models': models,
        'total_models': len(models)
    }


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@handle_api_errors(error_code='HEALTH_CHECK_ERROR', include_traceback=False)
def health_check_endpoint(model_dir: str = 'models') -> Dict[str, Any]:
    """
    Health check endpoint
    
    Args:
        model_dir: Model directory
    
    Returns:
        HealthCheckResponse dictionary
    """
    # Check model availability
    models_available = {
        'FCC': False,
        'CCR': False
    }
    
    try:
        validate_model_exists(model_dir, 'FCC')
        models_available['FCC'] = True
    except:
        pass
    
    try:
        validate_model_exists(model_dir, 'CCR')
        models_available['CCR'] = True
    except:
        pass
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Determine status
    if all(models_available.values()) and all(dependencies.values()):
        status = 'healthy'
        message = 'All systems operational'
    elif any(models_available.values()) and sum(dependencies.values()) >= 6:
        status = 'degraded'
        message = 'Some models or dependencies unavailable'
    else:
        status = 'unhealthy'
        message = 'Critical components missing'
    
    return {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'models_available': models_available,
        'dependencies': dependencies,
        'version': '2.0.0',
        'message': message
    }


if __name__ == '__main__':
    print("API Endpoint Implementations")
    print("\nAvailable endpoints:")
    print("  Training: train_model_endpoint")
    print("  Prediction: predict_endpoint, batch_predict_endpoint")
    print("  Submission: generate_submission_endpoint")
    print("  Quality: check_data_quality_endpoint")
    print("  Management: list_models_endpoint, health_check_endpoint")
