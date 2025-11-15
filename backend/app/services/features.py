"""
Feature Engineering Module for ANCAP DataChallenge 2025
Consolidates feature creation from multiple old modules

This module combines functionality from:
- feature_engineer.py
- gas_feature_engineer.py  
- h2_specialized_features.py

Key Features:
- Time-series features (lags, rolling stats, rate of change)
- Physics-informed features for refinery processes
- Gas composition features
- Automatic feature type detection
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import Numba for speed optimization
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_features(df: pd.DataFrame, feature_config: Optional[dict] = None) -> pd.DataFrame:
    """
    Comprehensive feature engineering with auto-detection
    
    Automatically detects data type and applies appropriate features:
    - Operational data: Time-series + physics-informed features
    - Gas composition: Gas chemistry features
    
    Args:
        df: Input DataFrame
        feature_config: Optional configuration dict with settings:
            - enable_time_series: bool (default True)
            - enable_physics: bool (default True)
            - enable_interactions: bool (default True)
            - lag_periods: list (default [1,2,3,4,6])
            - window_sizes: list (default [2,4,6,12])
    
    Returns:
        DataFrame with engineered features
    """
    if feature_config is None:
        feature_config = {
            'enable_time_series': True,
            'enable_physics': True,
            'enable_interactions': True,
            'lag_periods': [1, 2, 3, 4, 6],
            'window_sizes': [2, 4, 6, 12]
        }
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    print(f"Input shape: {df.shape}")
    
    # Detect data type
    if _is_gas_composition_data(df):
        print("📊 Detected: Gas composition data")
        df_features = create_gas_features(df, feature_config)
    else:
        print("📊 Detected: Operational data")
        df_features = create_operational_features(df, feature_config)
    
    print(f"Output shape: {df_features.shape}")
    print(f"Created {df_features.shape[1] - df.shape[1]} new features")
    print("="*60 + "\n")
    
    return df_features


def _is_gas_composition_data(df: pd.DataFrame) -> bool:
    """Check if DataFrame contains gas composition data"""
    gas_indicators = ['H2', 'C1', 'C2', 'C3', 'CH4']
    return any(col in df.columns for col in gas_indicators)


# ============================================================================
# OPERATIONAL FEATURES
# ============================================================================

def create_operational_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create physics-informed features from operational data
    
    Features include:
    1. Time-series features (lags, rolling stats, rate of change)
    2. Temperature features (gradients, ratios, efficiency)
    3. Pressure features (drops, ratios, flow resistance)
    4. Interaction features (cross-products of key variables)
    
    Args:
        df: DataFrame with operational columns
        config: Feature configuration
    
    Returns:
        DataFrame with engineered features
    """
    df_features = df.copy()
    
    # Identify column types
    temp_cols = [c for c in df.columns if any(x in c.lower() for x in ['temp', 'temperatura'])]
    pressure_cols = [c for c in df.columns if any(x in c.lower() for x in ['pres', 'presion', 'pressure'])]
    flow_cols = [c for c in df.columns if any(x in c.lower() for x in ['flow', 'flujo', 'caudal'])]
    
    print(f"  Found {len(temp_cols)} temperature, {len(pressure_cols)} pressure, {len(flow_cols)} flow columns")
    
    # 1. Time-series features
    if config['enable_time_series']:
        print("  Creating time-series features...")
        all_cols = temp_cols + pressure_cols + flow_cols
        df_features = create_time_series_features(
            df_features, all_cols, 
            lag_periods=config['lag_periods'],
            window_sizes=config['window_sizes']
        )
    
    # 2. Physics-informed features
    if config['enable_physics']:
        print("  Creating physics-informed features...")
        df_features = create_temperature_features(df_features, temp_cols)
        df_features = create_pressure_features(df_features, pressure_cols)
        df_features = create_efficiency_features(df_features, temp_cols, pressure_cols, flow_cols)
    
    # 3. Interaction features
    if config['enable_interactions']:
        print("  Creating interaction features...")
        df_features = create_interaction_features(df_features, temp_cols, pressure_cols, flow_cols)
    
    # Fill NaN values
    df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_features


def create_time_series_features(df: pd.DataFrame, columns: List[str],
                                lag_periods: List[int] = [1, 2, 3, 4, 6],
                                window_sizes: List[int] = [2, 4, 6, 12]) -> pd.DataFrame:
    """
    Create time-series features (lags, rolling stats, rate of change)
    
    Args:
        df: Input DataFrame
        columns: Columns to create features for
        lag_periods: Lag periods in hours
        window_sizes: Rolling window sizes in hours
    
    Returns:
        DataFrame with time-series features
    """
    df_ts = df.copy()
    
    # Filter to numeric columns only
    numeric_cols = [c for c in columns if c in df.columns and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if len(numeric_cols) == 0:
        return df_ts
    
    # Limit to most important columns to avoid feature explosion
    if len(numeric_cols) > 10:
        numeric_cols = numeric_cols[:10]
    
    for col in numeric_cols:
        # Lag features
        for lag in lag_periods:
            df_ts[f'{col}_lag{lag}h'] = df_ts[col].shift(lag)
        
        # Rolling statistics
        for window in window_sizes:
            if NUMBA_AVAILABLE:
                # Use optimized Numba functions
                col_values = df_ts[col].values
                df_ts[f'{col}_roll{window}h_mean'] = _rolling_mean_numba(col_values, window)
                df_ts[f'{col}_roll{window}h_std'] = _rolling_std_numba(col_values, window)
                df_ts[f'{col}_roll{window}h_min'] = _rolling_min_numba(col_values, window)
                df_ts[f'{col}_roll{window}h_max'] = _rolling_max_numba(col_values, window)
            else:
                # Use pandas (slower)
                df_ts[f'{col}_roll{window}h_mean'] = df_ts[col].rolling(window=window, min_periods=1).mean()
                df_ts[f'{col}_roll{window}h_std'] = df_ts[col].rolling(window=window, min_periods=1).std()
                df_ts[f'{col}_roll{window}h_min'] = df_ts[col].rolling(window=window, min_periods=1).min()
                df_ts[f'{col}_roll{window}h_max'] = df_ts[col].rolling(window=window, min_periods=1).max()
        
        # Rate of change
        df_ts[f'{col}_delta1h'] = df_ts[col].diff(1)
        df_ts[f'{col}_delta4h'] = df_ts[col].diff(4)
    
    return df_ts


# Numba-optimized rolling functions
@njit
def _rolling_mean_numba(arr, window):
    """Fast rolling mean with Numba"""
    n = len(arr)
    result = np.empty(n)
    for i in prange(n):
        start = max(0, i - window + 1)
        result[i] = np.mean(arr[start:i+1])
    return result


@njit
def _rolling_std_numba(arr, window):
    """Fast rolling std with Numba"""
    n = len(arr)
    result = np.empty(n)
    for i in prange(n):
        start = max(0, i - window + 1)
        result[i] = np.std(arr[start:i+1])
    return result


@njit
def _rolling_min_numba(arr, window):
    """Fast rolling min with Numba"""
    n = len(arr)
    result = np.empty(n)
    for i in prange(n):
        start = max(0, i - window + 1)
        result[i] = np.min(arr[start:i+1])
    return result


@njit
def _rolling_max_numba(arr, window):
    """Fast rolling max with Numba"""
    n = len(arr)
    result = np.empty(n)
    for i in prange(n):
        start = max(0, i - window + 1)
        result[i] = np.max(arr[start:i+1])
    return result


def create_temperature_features(df: pd.DataFrame, temp_cols: List[str]) -> pd.DataFrame:
    """
    Create temperature-based features
    
    Features:
    - Temperature gradients (differences between zones)
    - Temperature ratios
    - Average/max/min temperatures
    
    Args:
        df: Input DataFrame
        temp_cols: Temperature column names
    
    Returns:
        DataFrame with temperature features
    """
    df_temp = df.copy()
    
    if len(temp_cols) < 2:
        return df_temp
    
    # Temperature statistics
    df_temp['temp_mean'] = df[temp_cols].mean(axis=1)
    df_temp['temp_max'] = df[temp_cols].max(axis=1)
    df_temp['temp_min'] = df[temp_cols].min(axis=1)
    df_temp['temp_range'] = df_temp['temp_max'] - df_temp['temp_min']
    df_temp['temp_std'] = df[temp_cols].std(axis=1)
    
    # Temperature gradients (pairwise differences)
    for i in range(min(3, len(temp_cols)-1)):  # Limit to avoid explosion
        for j in range(i+1, min(3, len(temp_cols))):
            col1, col2 = temp_cols[i], temp_cols[j]
            df_temp[f'temp_gradient_{i}_{j}'] = df[col1] - df[col2]
    
    return df_temp


def create_pressure_features(df: pd.DataFrame, pressure_cols: List[str]) -> pd.DataFrame:
    """
    Create pressure-based features
    
    Features:
    - Pressure drops (differences)
    - Pressure ratios
    - Average/max/min pressures
    
    Args:
        df: Input DataFrame
        pressure_cols: Pressure column names
    
    Returns:
        DataFrame with pressure features
    """
    df_pres = df.copy()
    
    if len(pressure_cols) < 2:
        return df_pres
    
    # Pressure statistics
    df_pres['pressure_mean'] = df[pressure_cols].mean(axis=1)
    df_pres['pressure_max'] = df[pressure_cols].max(axis=1)
    df_pres['pressure_min'] = df[pressure_cols].min(axis=1)
    df_pres['pressure_range'] = df_pres['pressure_max'] - df_pres['pressure_min']
    
    # Pressure drops (pairwise)
    for i in range(min(2, len(pressure_cols)-1)):
        for j in range(i+1, min(2, len(pressure_cols))):
            col1, col2 = pressure_cols[i], pressure_cols[j]
            df_pres[f'pressure_drop_{i}_{j}'] = df[col1] - df[col2]
    
    return df_pres


def create_efficiency_features(df: pd.DataFrame, temp_cols: List[str],
                               pressure_cols: List[str], flow_cols: List[str]) -> pd.DataFrame:
    """
    Create process efficiency features
    
    Args:
        df: Input DataFrame
        temp_cols: Temperature columns
        pressure_cols: Pressure columns
        flow_cols: Flow columns
    
    Returns:
        DataFrame with efficiency features
    """
    df_eff = df.copy()
    
    # Temperature efficiency (ratio of extremes)
    if len(temp_cols) >= 2:
        temp_max = df[temp_cols].max(axis=1)
        temp_min = df[temp_cols].min(axis=1)
        df_eff['temp_efficiency'] = (temp_max - temp_min) / (temp_max + 1e-6)
    
    # Pressure efficiency
    if len(pressure_cols) >= 2:
        pres_max = df[pressure_cols].max(axis=1)
        pres_min = df[pressure_cols].min(axis=1)
        df_eff['pressure_efficiency'] = (pres_max - pres_min) / (pres_max + 1e-6)
    
    # Flow stability (coefficient of variation)
    if len(flow_cols) >= 2:
        flow_mean = df[flow_cols].mean(axis=1)
        flow_std = df[flow_cols].std(axis=1)
        df_eff['flow_stability'] = flow_std / (flow_mean + 1e-6)
    
    return df_eff


def create_interaction_features(df: pd.DataFrame, temp_cols: List[str],
                                pressure_cols: List[str], flow_cols: List[str]) -> pd.DataFrame:
    """
    Create interaction features between different variable types
    
    Args:
        df: Input DataFrame
        temp_cols: Temperature columns
        pressure_cols: Pressure columns
        flow_cols: Flow columns
    
    Returns:
        DataFrame with interaction features
    """
    df_int = df.copy()
    
    # Temperature × Pressure interactions (select first of each)
    if temp_cols and pressure_cols:
        temp_col = temp_cols[0]
        pres_col = pressure_cols[0]
        df_int[f'{temp_col}_x_{pres_col}'] = df[temp_col] * df[pres_col]
        df_int[f'{temp_col}_div_{pres_col}'] = df[temp_col] / (df[pres_col] + 1e-6)
    
    # Temperature × Flow interactions
    if temp_cols and flow_cols:
        temp_col = temp_cols[0]
        flow_col = flow_cols[0]
        df_int[f'{temp_col}_x_{flow_col}'] = df[temp_col] * df[flow_col]
    
    # Pressure × Flow interactions
    if pressure_cols and flow_cols:
        pres_col = pressure_cols[0]
        flow_col = flow_cols[0]
        df_int[f'{pres_col}_x_{flow_col}'] = df[pres_col] * df[flow_col]
    
    return df_int


# ============================================================================
# GAS COMPOSITION FEATURES
# ============================================================================

def create_gas_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create features from gas composition data
    
    Features include:
    - Molecular weight calculations
    - Gas ratios (H2/C1, C2/C3, etc.)
    - Heating value estimations
    - Component interactions
    
    Args:
        df: DataFrame with gas composition columns
        config: Feature configuration
    
    Returns:
        DataFrame with gas features
    """
    df_gas = df.copy()
    
    print("  Creating gas composition features...")
    
    # Identify gas components
    gas_components = ['H2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CH4', 'CO', 'CO2', 'N2']
    available_components = [col for col in gas_components if col in df.columns]
    
    print(f"    Found {len(available_components)} gas components")
    
    # 1. Gas ratios
    if 'H2' in df.columns and 'C1' in df.columns:
        df_gas['H2_C1_ratio'] = df['H2'] / (df['C1'] + 1e-6)
    
    if 'C2' in df.columns and 'C3' in df.columns:
        df_gas['C2_C3_ratio'] = df['C2'] / (df['C3'] + 1e-6)
    
    # 2. Total hydrocarbons
    hydrocarbon_cols = [c for c in available_components if c.startswith('C') or c == 'CH4']
    if hydrocarbon_cols:
        df_gas['total_hydrocarbons'] = df[hydrocarbon_cols].sum(axis=1)
    
    # 3. Hydrogen content indicators
    if 'H2' in df.columns:
        df_gas['H2_squared'] = df['H2'] ** 2
        df_gas['H2_log'] = np.log1p(df['H2'])
    
    # 4. Component statistics
    if len(available_components) >= 3:
        df_gas['gas_mean'] = df[available_components].mean(axis=1)
        df_gas['gas_std'] = df[available_components].std(axis=1)
        df_gas['gas_max'] = df[available_components].max(axis=1)
    
    # 5. Time-series features if enabled
    if config.get('enable_time_series', True):
        df_gas = create_time_series_features(
            df_gas, available_components,
            lag_periods=[1, 2, 3],
            window_sizes=[2, 4, 6]
        )
    
    return df_gas


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to be compatible with all ML libraries
    
    Removes special characters that can cause issues with:
    - LightGBM
    - CatBoost
    - AutoGluon
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    
    # Replace special characters
    new_columns = []
    for col in df_clean.columns:
        # Replace problematic characters
        clean_col = str(col)
        for char in ['(', ')', '[', ']', '{', '}', '"', "'", ':', ',', '%', '/', '\\', ' ']:
            clean_col = clean_col.replace(char, '_')
        
        # Remove consecutive underscores
        while '__' in clean_col:
            clean_col = clean_col.replace('__', '_')
        
        # Remove leading/trailing underscores
        clean_col = clean_col.strip('_')
        
        new_columns.append(clean_col)
    
    df_clean.columns = new_columns
    
    return df_clean


def select_important_features(df: pd.DataFrame, target: pd.Series,
                              n_features: int = 100) -> List[str]:
    """
    Select most important features using correlation
    
    Args:
        df: Feature DataFrame
        target: Target variable
        n_features: Number of features to select
    
    Returns:
        List of selected feature names
    """
    # Compute correlations
    correlations = df.corrwith(target).abs().sort_values(ascending=False)
    
    # Select top N
    selected = correlations.head(n_features).index.tolist()
    
    print(f"  Selected {len(selected)} most important features")
    
    return selected


if __name__ == '__main__':
    print("Feature Engineering Module")
    print("Import this module to use: from backend.app.services.features import create_features")
