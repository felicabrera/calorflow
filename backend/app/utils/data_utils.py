"""
Data Utilities Module for ANCAP DataChallenge 2025
Handles data loading, preprocessing, and quality checks

This module combines functionality from:
- data_loader.py
- data_quality.py
- preprocess_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath: str, sep: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV file with automatic separator detection
    
    Args:
        filepath: Path to CSV file
        sep: Separator (if None, auto-detect)
    
    Returns:
        Loaded DataFrame
    """
    print(f"Loading data from: {filepath}")
    
    if sep is None:
        # Try different separators
        separators = [';', ',', '\t', '|']
        df = None
        
        for separator in separators:
            try:
                df = pd.read_csv(filepath, sep=separator)
                if len(df.columns) > 1:
                    print(f"  Loaded with separator '{separator}': {len(df.columns)} columns")
                    break
            except Exception as e:
                continue
        
        if df is None or len(df.columns) <= 1:
            raise ValueError(f"Could not load data from {filepath}")
    else:
        df = pd.read_csv(filepath, sep=sep)
        print(f"  Loaded {len(df.columns)} columns, {len(df)} rows")
    
    return df


def load_fcc_data(data_dir: str = 'data/FCC - Cracking Catalítico') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FCC (Fluid Catalytic Cracking) process data
    
    Args:
        data_dir: Directory containing FCC data files
    
    Returns:
        Tuple of (operational_data, gas_composition_data)
    """
    print("\n" + "="*60)
    print("LOADING FCC DATA")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Load operational data (Predictoras files)
    operational_files = list(data_path.glob('Predictoras*FCC*.csv'))
    if operational_files:
        print(f"  Found {len(operational_files)} operational file(s)")
        # Load and concatenate all operational files
        dfs = [load_data(str(f)) for f in operational_files]
        df_operational = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        print("  Warning: No operational data file found (looking for Predictoras*FCC*.csv)")
        df_operational = pd.DataFrame()
    
    # Load gas composition data (R-CRACKING files)
    gas_files = list(data_path.glob('R-CRACKING*.csv'))
    if gas_files:
        print(f"  Found {len(gas_files)} gas composition file(s)")
        # Load and concatenate all gas files
        dfs = [load_data(str(f)) for f in gas_files]
        df_gas = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        print("  Warning: No gas composition file found (looking for R-CRACKING*.csv)")
        df_gas = pd.DataFrame()
    
    print("="*60 + "\n")
    
    return df_operational, df_gas


def load_ccr_data(data_dir: str = 'data/CCR - Reforming Catalítico') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CCR (Catalytic Reforming) process data
    
    Args:
        data_dir: Directory containing CCR data files
    
    Returns:
        Tuple of (operational_data, gas_composition_data)
    """
    print("\n" + "="*60)
    print("LOADING CCR DATA")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Load operational data (Predictoras files)
    operational_files = list(data_path.glob('Predictoras*CCR*.csv'))
    if operational_files:
        print(f"  Found {len(operational_files)} operational file(s)")
        # Load and concatenate all operational files
        dfs = [load_data(str(f)) for f in operational_files]
        df_operational = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        print("  Warning: No operational data file found (looking for Predictoras*CCR*.csv)")
        df_operational = pd.DataFrame()
    
    # Load gas composition data (R-RFM files)
    gas_files = list(data_path.glob('*RFM*.csv')) + list(data_path.glob('*rfm*.csv'))
    if gas_files:
        print(f"  Found {len(gas_files)} gas composition file(s)")
        # Load and concatenate all gas files
        dfs = [load_data(str(f)) for f in gas_files]
        df_gas = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        print("  Warning: No gas composition file found (looking for *RFM*.csv or *rfm*.csv)")
        df_gas = pd.DataFrame()
    
    print("="*60 + "\n")
    
    return df_operational, df_gas


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df: pd.DataFrame, 
                   handle_missing: str = 'interpolate',
                   remove_outliers: bool = True,
                   outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Preprocess data with cleaning and quality checks
    
    Args:
        df: Input DataFrame
        handle_missing: Method to handle missing values ('drop', 'interpolate', 'ffill')
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Preprocessed DataFrame
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    print(f"Input shape: {df.shape}")
    
    df_processed = df.copy()
    
    # 1. Handle missing values
    print(f"\n1. Handling missing values (method: {handle_missing})...")
    missing_before = df_processed.isnull().sum().sum()
    
    if handle_missing == 'drop':
        df_processed = df_processed.dropna()
    elif handle_missing == 'interpolate':
        df_processed = df_processed.interpolate(method='linear', limit_direction='both')
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    elif handle_missing == 'ffill':
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df_processed.isnull().sum().sum()
    print(f"   Missing values: {missing_before} → {missing_after}")
    
    # 2. Remove outliers
    if remove_outliers:
        print(f"\n2. Removing outliers (threshold: {outlier_threshold} std)...")
        rows_before = len(df_processed)
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['PCI', 'H2']:  # Don't remove outliers from targets
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / (df_processed[col].std() + 1e-10))
                df_processed = df_processed[z_scores < outlier_threshold]
        
        rows_after = len(df_processed)
        print(f"   Rows removed: {rows_before - rows_after} ({(rows_before - rows_after) / rows_before * 100:.2f}%)")
    
    # 3. Convert data types
    print("\n3. Optimizing data types...")
    for col in df_processed.select_dtypes(include=['float64']).columns:
        df_processed[col] = df_processed[col].astype('float32')
    
    print(f"\nOutput shape: {df_processed.shape}")
    print("="*60 + "\n")
    
    return df_processed


def merge_operational_and_gas(df_operational: pd.DataFrame, 
                              df_gas: pd.DataFrame,
                              timestamp_col: str = 'timestamp',
                              tolerance: str = '1H') -> pd.DataFrame:
    """
    Merge operational and gas composition data
    
    Args:
        df_operational: Operational data
        df_gas: Gas composition data
        timestamp_col: Name of timestamp column
        tolerance: Time tolerance for merging
    
    Returns:
        Merged DataFrame
    """
    print("Merging operational and gas composition data...")
    
    if timestamp_col not in df_operational.columns or timestamp_col not in df_gas.columns:
        print("  Warning: Timestamp column not found, performing simple merge")
        return pd.concat([df_operational, df_gas], axis=1)
    
    # Convert timestamps
    df_operational[timestamp_col] = pd.to_datetime(df_operational[timestamp_col])
    df_gas[timestamp_col] = pd.to_datetime(df_gas[timestamp_col])
    
    # Merge with tolerance
    df_merged = pd.merge_asof(
        df_operational.sort_values(timestamp_col),
        df_gas.sort_values(timestamp_col),
        on=timestamp_col,
        tolerance=pd.Timedelta(tolerance),
        direction='nearest'
    )
    
    print(f"  Merged: {len(df_operational)} operational + {len(df_gas)} gas -> {len(df_merged)} rows")
    
    return df_merged


# ============================================================================
# DATA QUALITY CHECKS
# ============================================================================

def check_data_quality(df: pd.DataFrame, target_cols: Optional[List[str]] = None) -> Dict:
    """
    Perform comprehensive data quality checks
    
    Args:
        df: DataFrame to check
        target_cols: Optional list of target columns to analyze separately
    
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {}
    
    # Basic statistics
    quality_report['n_samples'] = len(df)
    quality_report['n_features'] = len(df.columns)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    quality_report['missing_values'] = {col: int(count) for col, count in missing.items() if count > 0}
    quality_report['total_missing_pct'] = float(missing_pct.mean())
    
    # Duplicates
    quality_report['duplicates'] = int(df.duplicated().sum())
    
    # Feature types
    quality_report['numeric_features'] = len(df.select_dtypes(include=[np.number]).columns)
    quality_report['categorical_features'] = len(df.select_dtypes(include=['object']).columns)
    quality_report['datetime_features'] = len(df.select_dtypes(include=['datetime64']).columns)
    
    return quality_report


def validate_train_test_split(X_train: pd.DataFrame, X_test: pd.DataFrame) -> bool:
    """
    Validate that train and test sets have compatible features
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        True if valid, False otherwise
    """
    print("Validating train/test split...")
    
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    # Check for missing columns
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    
    if missing_in_test:
        print(f"  Warning: Columns in train but not in test: {missing_in_test}")
        return False
    
    if missing_in_train:
        print(f"  Warning: Columns in test but not in train: {missing_in_train}")
        return False
    
    # Check column order
    if list(X_train.columns) != list(X_test.columns):
        print("  Warning: Column order differs between train and test")
        return False
    
    print("  Train/test split is valid")
    return True


# ============================================================================
# DATA SPLITTING
# ============================================================================

def prepare_train_test_split(df: pd.DataFrame,
                             target_cols: List[str] = ['PCI', 'H2'],
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train/test split with validation
    
    Args:
        df: Input DataFrame
        target_cols: List of target column names
        test_size: Fraction of data for testing
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    print(f"\nPreparing train/test split (test_size={test_size})...")
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    X = df[feature_cols]
    y = df[target_cols]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols)}")
    
    # Validate
    validate_train_test_split(X_train, X_test)
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to CSV"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filepath: str) -> pd.DataFrame:
    """Load processed data from CSV"""
    df = pd.read_csv(filepath)
    print(f"Loaded processed data from {filepath}")
    return df


if __name__ == '__main__':
    print("Data Utilities Module")
    print("Import this module to use: from src.data_utils import load_data, preprocess_data")
