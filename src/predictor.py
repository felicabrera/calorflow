"""
Prediction Module for ANCAP DataChallenge 2025
Handles inference using trained models and generates competition submissions

This module consolidates prediction functionality from:
- predict_competition.py
- predict_test.py
- predict_ensemble.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def load_trained_models(model_dir: str = 'models', 
                       process_name: str = 'FCC') -> Tuple[Any, Any]:
    """
    Load trained ensemble models for predictions
    
    Args:
        model_dir: Directory containing saved models
        process_name: Name of process (FCC/CCR)
    
    Returns:
        Tuple of (pci_ensemble, h2_ensemble)
    """
    from src.trainer import MLTrainer
    
    print(f"\nLoading trained models for {process_name}...")
    
    model_path = Path(model_dir)
    
    # Try to load full trainer
    try:
        trainer = MLTrainer.load(model_dir, process_name)
        print(f"  ✅ Loaded full trainer from {model_path}")
        return trainer.ensemble_pci, trainer.ensemble_h2
    except Exception as e:
        print(f"  ⚠️  Could not load full trainer: {e}")
        
        # Try to load individual ensembles
        try:
            pci_ensemble = joblib.load(model_path / f'{process_name}_ensemble_pci.joblib')
            h2_ensemble = joblib.load(model_path / f'{process_name}_ensemble_h2.joblib')
            print(f"  ✅ Loaded individual ensembles from {model_path}")
            return pci_ensemble, h2_ensemble
        except Exception as e2:
            raise FileNotFoundError(
                f"Could not load models for {process_name} from {model_path}\n"
                f"Please train models first: python train.py"
            )


def predict_from_features(X: pd.DataFrame, 
                         model_dir: str = 'models',
                         process_name: str = 'FCC') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from features using trained models
    
    Args:
        X: Feature DataFrame
        model_dir: Directory containing saved models
        process_name: Process name (FCC/CCR)
    
    Returns:
        Tuple of (pci_predictions, h2_predictions)
    """
    print("\n" + "="*80)
    print(f"GENERATING PREDICTIONS - {process_name}")
    print("="*80)
    print(f"Samples: {len(X)}")
    print(f"Features: {len(X.columns)}")
    
    # Load models
    pci_ensemble, h2_ensemble = load_trained_models(model_dir, process_name)
    
    # Generate predictions
    print("\nPredicting...")
    y_pred_pci = pci_ensemble.predict(X)
    y_pred_h2 = h2_ensemble.predict(X)
    
    # Show statistics
    print(f"\nPrediction Statistics:")
    print(f"  PCI: min={y_pred_pci.min():.2f}, max={y_pred_pci.max():.2f}, mean={y_pred_pci.mean():.2f}")
    print(f"  H2:  min={y_pred_h2.min():.2f}, max={y_pred_h2.max():.2f}, mean={y_pred_h2.mean():.2f}")
    print("="*80 + "\n")
    
    return y_pred_pci, y_pred_h2


def predict_from_raw_data(data_path: str,
                          model_dir: str = 'models',
                          process_name: str = 'FCC',
                          feature_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Complete prediction pipeline from raw data to predictions
    
    Args:
        data_path: Path to raw data CSV
        model_dir: Directory containing saved models
        process_name: Process name (FCC/CCR)
        feature_config: Optional feature engineering configuration
    
    Returns:
        DataFrame with predictions
    """
    from src.data_utils import load_data, preprocess_data
    from src.features import create_features, clean_feature_names
    
    print("\n" + "="*80)
    print(f"COMPLETE PREDICTION PIPELINE - {process_name}")
    print("="*80)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    df = load_data(data_path)
    
    # 2. Preprocess
    print("\n[2/4] Preprocessing...")
    df = preprocess_data(df, handle_missing='interpolate', remove_outliers=False)
    
    # 3. Create features
    print("\n[3/4] Creating features...")
    df = create_features(df, feature_config)
    df = clean_feature_names(df)
    
    # 4. Predict
    print("\n[4/4] Generating predictions...")
    y_pred_pci, y_pred_h2 = predict_from_features(df, model_dir, process_name)
    
    # Add predictions to DataFrame
    result = df.copy()
    result['PCI_pred'] = y_pred_pci
    result['H2_pred'] = y_pred_h2
    
    print("✅ Prediction pipeline completed!")
    
    return result


# ============================================================================
# COMPETITION SUBMISSION
# ============================================================================

def load_test_data_for_prediction(process: str = 'FCC',
                                  data_dir: str = 'data') -> pd.DataFrame:
    """
    Load test data for competition predictions
    
    Args:
        process: Process name (FCC/CCR)
        data_dir: Root data directory
    
    Returns:
        DataFrame with test features
    """
    from src.data_utils import load_fcc_data, load_ccr_data, merge_operational_and_gas, preprocess_data
    from src.features import create_features, clean_feature_names
    
    print(f"\nLoading test data for {process}...")
    
    # Load data based on process
    if process == 'FCC':
        df_operational, df_gas = load_fcc_data(f'{data_dir}/FCC - Cracking Catalítico')
    elif process == 'CCR':
        df_operational, df_gas = load_ccr_data(f'{data_dir}/CCR - Reforming Catalítico')
    else:
        raise ValueError(f"Unknown process: {process}")
    
    # Use operational data (test files don't have gas composition - that's what we predict!)
    if not df_operational.empty:
        df = df_operational
    else:
        raise ValueError(f"No operational data found for {process}")
    
    # Filter for test period (202503-202508)
    if 'sampled_date' in df.columns or 'timestamp' in df.columns:
        date_col = 'sampled_date' if 'sampled_date' in df.columns else 'timestamp'
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Filter for test period
        test_start = pd.Timestamp('2025-03-01')
        test_end = pd.Timestamp('2025-08-31')
        df = df[(df[date_col] >= test_start) & (df[date_col] <= test_end)]
        print(f"  ✅ Filtered to test period: {len(df)} samples")
    
    # Preprocess and create features
    df = preprocess_data(df, handle_missing='interpolate', remove_outliers=False)
    df = create_features(df)
    df = clean_feature_names(df)
    
    return df


def generate_competition_submission(process: str = 'FCC',
                                   model_dir: str = 'models',
                                   data_dir: str = 'data',
                                   output_dir: str = 'predictions') -> str:
    """
    Generate competition submission file
    
    Args:
        process: Process name (FCC/CCR)
        model_dir: Directory containing trained models
        data_dir: Directory containing data
        output_dir: Directory to save predictions
    
    Returns:
        Path to saved submission file
    """
    print("\n" + "="*80)
    print(f"GENERATING COMPETITION SUBMISSION - {process}")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("\n[1/3] Loading test data...")
    X_test = load_test_data_for_prediction(process, data_dir)
    
    # Generate predictions
    print("\n[2/3] Generating predictions...")
    y_pred_pci, y_pred_h2 = predict_from_features(X_test, model_dir, process)
    
    # Create submission DataFrame
    print("\n[3/3] Creating submission file...")
    submission = pd.DataFrame({
        'PCI': y_pred_pci,
        'H2': y_pred_h2
    })
    
    # Add timestamp if available
    if 'sampled_date' in X_test.columns:
        submission.insert(0, 'sampled_date', X_test['sampled_date'].values)
    elif 'timestamp' in X_test.columns:
        submission.insert(0, 'sampled_date', X_test['timestamp'].values)
    
    # Save submission
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_file = output_path / f'{process}_submission_{timestamp}.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Submission saved: {submission_file}")
    print(f"   Samples: {len(submission)}")
    print(f"   PCI range: [{submission['PCI'].min():.2f}, {submission['PCI'].max():.2f}]")
    print(f"   H2 range: [{submission['H2'].min():.2f}, {submission['H2'].max():.2f}]")
    print("="*80 + "\n")
    
    return str(submission_file)


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_multiple_files(file_paths: list,
                          model_dir: str = 'models',
                          process_name: str = 'FCC',
                          output_dir: str = 'predictions') -> list:
    """
    Generate predictions for multiple files
    
    Args:
        file_paths: List of file paths to predict
        model_dir: Directory containing models
        process_name: Process name (FCC/CCR)
        output_dir: Directory to save predictions
    
    Returns:
        List of output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    
    for file_path in file_paths:
        print(f"\nProcessing: {file_path}")
        
        # Generate predictions
        result = predict_from_raw_data(file_path, model_dir, process_name)
        
        # Save predictions
        input_name = Path(file_path).stem
        output_file = output_path / f'{input_name}_predictions.csv'
        result.to_csv(output_file, index=False)
        
        print(f"  ✅ Saved: {output_file}")
        output_files.append(str(output_file))
    
    return output_files


# ============================================================================
# EVALUATION (when targets are available)
# ============================================================================

def evaluate_predictions(y_true_pci: np.ndarray, y_pred_pci: np.ndarray,
                        y_true_h2: np.ndarray, y_pred_h2: np.ndarray) -> Dict[str, float]:
    """
    Evaluate predictions when true values are available
    
    Args:
        y_true_pci: True PCI values
        y_pred_pci: Predicted PCI values
        y_true_h2: True H2 values
        y_pred_h2: Predicted H2 values
    
    Returns:
        Dictionary with evaluation metrics
    """
    from src.trainer import compute_metrics, compute_competition_score
    
    print("\n" + "="*80)
    print("PREDICTION EVALUATION")
    print("="*80)
    
    # Compute metrics
    metrics_pci = compute_metrics(y_true_pci, y_pred_pci, 'PCI')
    metrics_h2 = compute_metrics(y_true_h2, y_pred_h2, 'H2')
    competition_score = compute_competition_score(metrics_pci, metrics_h2)
    
    # Display results
    print(f"\nPCI Metrics:")
    print(f"  RMSE: {metrics_pci['PCI_rmse']:.4f}")
    print(f"  MAE:  {metrics_pci['PCI_mae']:.4f}")
    print(f"  R²:   {metrics_pci['PCI_r2']:.4f}")
    print(f"  Within ±10%: {metrics_pci['PCI_within_10pct']} ({metrics_pci['PCI_within_10pct_ratio']*100:.1f}%)")
    
    print(f"\nH2 Metrics:")
    print(f"  RMSE: {metrics_h2['H2_rmse']:.4f}")
    print(f"  MAE:  {metrics_h2['H2_mae']:.4f}")
    print(f"  R²:   {metrics_h2['H2_r2']:.4f}")
    print(f"  Within ±10%: {metrics_h2['H2_within_10pct']} ({metrics_h2['H2_within_10pct_ratio']*100:.1f}%)")
    
    print(f"\nCompetition Score:")
    print(f"  RMSE_prom: {competition_score['rmse_prom']:.4f}")
    print(f"  Within ±10% Total: {competition_score['within_10_pci'] + competition_score['within_10_h2']}")
    print("="*80 + "\n")
    
    return {**metrics_pci, **metrics_h2, **competition_score}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_predict(process: str = 'FCC') -> str:
    """
    Quick prediction for competition submission
    
    Args:
        process: Process name (FCC/CCR)
    
    Returns:
        Path to submission file
    """
    return generate_competition_submission(
        process=process,
        model_dir='models',
        data_dir='data',
        output_dir='predictions'
    )


if __name__ == '__main__':
    print("Prediction Module for ANCAP DataChallenge 2025")
    print("\nUsage:")
    print("  from src.predictor import quick_predict, generate_competition_submission")
    print("  quick_predict('FCC')  # Generate FCC predictions")
    print("  quick_predict('CCR')  # Generate CCR predictions")
