"""
ANCAP 2025 DataChallenge - Consolidated ML Package

This package provides a streamlined, production-ready ML pipeline for
predicting PCI (Power Calorific Index) and H2 (Hydrogen content) from
refinery process data.

Main modules:
- trainer: Unified training, optimization, and inference
- features: Feature engineering (time-series, physics-informed, gas composition)
- data_utils: Data loading, preprocessing, and quality checks

Quick Start:
-----------
from src.trainer import MLTrainer, TrainerConfig, train_pipeline
from src.data_utils import load_data, preprocess_data
from src.features import create_features

# Load and prepare data
df = load_data('data/your_data.csv')
df = preprocess_data(df)
df = create_features(df)

# Train models
X = df.drop(['PCI', 'H2'], axis=1)
y_pci = df['PCI']
y_h2 = df['H2']

trainer = train_pipeline(X, y_pci, y_h2, process_name='FCC', output_dir='models')

# Make predictions
y_pred_pci, y_pred_h2 = trainer.predict(X_test)
"""

__version__ = '2.0.0'
__author__ = 'ANCAP Team'

from .trainer import MLTrainer, TrainerConfig, train_pipeline, predict_with_models
from .features import create_features, clean_feature_names
from .data_utils import (
    load_data, 
    load_fcc_data, 
    load_ccr_data,
    preprocess_data,
    check_data_quality
)
from .predictor import (
    load_trained_models,
    predict_from_features,
    predict_from_raw_data,
    generate_competition_submission,
    quick_predict
)
from .api import (
    train_model_endpoint,
    predict_endpoint,
    batch_predict_endpoint,
    generate_submission_endpoint,
    check_data_quality_endpoint,
    list_models_endpoint,
    health_check_endpoint,
    setup_logging,
    get_logger
)

__all__ = [
    # Trainer
    'MLTrainer',
    'TrainerConfig', 
    'train_pipeline',
    'predict_with_models',
    
    # Predictor
    'load_trained_models',
    'predict_from_features',
    'predict_from_raw_data',
    'generate_competition_submission',
    'quick_predict',
    
    # Features
    'create_features',
    'clean_feature_names',
    
    # Data utilities
    'load_data',
    'load_fcc_data',
    'load_ccr_data',
    'preprocess_data',
    'check_data_quality',
    
    # API Endpoints
    'train_model_endpoint',
    'predict_endpoint',
    'batch_predict_endpoint',
    'generate_submission_endpoint',
    'check_data_quality_endpoint',
    'list_models_endpoint',
    'health_check_endpoint',
    
    # Logging
    'setup_logging',
    'get_logger',
]
