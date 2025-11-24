"""
Lightweight port of data preprocessing functions from the `train_competition.ipynb` notebook.
This module keeps the essential steps so the API server can preprocess and prepare datasets
for training and prediction.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Short PCI mapping used by the notebook
PCI_VALUES = {
    'H2': 2580, 'C1': 8590, 'C2': 14320, 'C2=': 13560,
    'C3': 20050, 'C3=': 18700, '= C3': 18700,
    'I-C4': 27500, 'N-C4': 28000, 'I=C4': 26400,
    '1=C4': 26400, 'C-2=C4': 26400, 'T-2=C4': 26400,
    '1,3=C4': 24900, 'I-C5': 33800, 'N-C5': 34200,
    'CO': 3020, 'CO2': 0, 'N2': 0, 'O2': 0, 'H2S': 5850,
}


def load_gas_composition(filepath: str) -> pd.DataFrame:
    """Load and pivot gas composition data, calculate PCI and ensure H2 column."""
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    if 'sampled_date' in df.columns:
        df['sampled_date'] = pd.to_datetime(df['sampled_date'], errors='coerce')
    else:
        df['sampled_date'] = pd.NaT

    df = df[df.get('analysis', '') == 'R-COMPONEN'].copy() if 'analysis' in df.columns else df
    df.loc[df['name'] == 'NULL', 'name'] = '= C3'
    df = df[~df['name'].isin(['Equipo', None])].copy()

    df['FORMATTED_ENTRY'] = df['FORMATTED_ENTRY'].astype(str).str.replace('<', '').str.replace('>', '').str.replace(',', '')
    df['FORMATTED_ENTRY'] = pd.to_numeric(df['FORMATTED_ENTRY'], errors='coerce')

    pivot_df = df.pivot_table(index='sampled_date', columns='name', values='FORMATTED_ENTRY', aggfunc='mean').reset_index()

    # Calculate PCI
    pci_values = []
    for idx, row in pivot_df.iterrows():
        pci = 0.0
        for comp, val in PCI_VALUES.items():
            if comp in pivot_df.columns:
                pci += val * (row.get(comp, 0) / 100.0)
        pci_values.append(pci)

    pivot_df['PCI'] = pci_values
    if 'H2' not in pivot_df.columns:
        pivot_df['H2'] = np.nan

    return pivot_df


def load_operational_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'fecha' in col.lower()]
    if date_cols:
        df['sampled_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    else:
        df['sampled_date'] = pd.NaT
    return df


def load_feedstock_properties(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    if 'sampled_date' in df.columns:
        df['sampled_date'] = pd.to_datetime(df['sampled_date'], errors='coerce')
    else:
        df['sampled_date'] = pd.NaT
    if 'FORMATTED_ENTRY' in df.columns:
        df['FORMATTED_ENTRY'] = df['FORMATTED_ENTRY'].astype(str).str.replace('<', '').str.replace('>', '').str.replace(',', '')
        df['FORMATTED_ENTRY'] = pd.to_numeric(df['FORMATTED_ENTRY'], errors='coerce')

    pivot_df = df.pivot_table(index='sampled_date', columns='name', values='FORMATTED_ENTRY', aggfunc='mean').reset_index()
    pivot_df.columns = ['sampled_date'] + [f'prop_{col}' for col in pivot_df.columns[1:]]
    return pivot_df


def merge_and_aggregate(gas_df: pd.DataFrame, operational_df: pd.DataFrame, feedstock_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge datasets with hourly aggregation and keep rows with actual gas measurements."""
    operational_df = operational_df.copy()
    operational_df['hour'] = operational_df['sampled_date'].dt.floor('1H')
    numeric_cols = operational_df.select_dtypes(include='number').columns.tolist()
    agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
    if not numeric_cols:
        # If no numeric cols, return just gas_df merged
        merged = gas_df.copy()
        merged['has_actual_measurement'] = merged['PCI'].notna()
        return merged

    operational_hourly = operational_df.groupby('hour').agg(agg_dict).reset_index()
    new_cols = ['sampled_date']
    for col in numeric_cols:
        new_cols.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max'])
    operational_hourly.columns = new_cols

    merged = operational_hourly.copy()
    gas_for_merge = gas_df[['sampled_date', 'PCI', 'H2']].copy()
    gas_for_merge['gas_hour'] = gas_for_merge['sampled_date'].dt.floor('1H')

    merged = merged.merge(gas_for_merge[['gas_hour', 'PCI', 'H2']], left_on='sampled_date', right_on='gas_hour', how='left').drop(columns=['gas_hour'], errors='ignore')
    merged['has_actual_measurement'] = merged['PCI'].notna()
    merged['sample_weight'] = merged['has_actual_measurement'].astype(float)

    if feedstock_df is not None:
        merged = pd.merge_asof(merged.sort_values('sampled_date'), feedstock_df.sort_values('sampled_date'), on='sampled_date', direction='nearest', tolerance=pd.Timedelta('6H'))

    merged = merged[merged['has_actual_measurement']].copy()
    merged = merged.dropna(subset=['PCI', 'H2'])
    return merged


def prepare_fcc_data(base_path: str = 'data/FCC - Cracking Catalítico'):
    base = Path(base_path)
    train_gas = load_gas_composition(str(base / 'R-CRACKING_402E_202406_202502.csv'))
    train_operational = load_operational_data(str(base / 'Predictoras_202406_202502_FCC.csv'))
    train_feedstock = load_feedstock_properties(str(base / 'R-CRACKING_CARGA_CRACKING_202406_202502.csv'))
    train_df = merge_and_aggregate(train_gas, train_operational, train_feedstock)

    test_timestamps = pd.read_csv(base / 'R-CRACKING_402E_202503_202508 - a estimar.csv', sep=';')
    test_timestamps['sampled_date'] = pd.to_datetime(test_timestamps['sampled_date'], errors='coerce')
    test_operational = load_operational_data(str(base / 'Predictoras_202503_202508_FCC.csv'))
    test_feedstock = load_feedstock_properties(str(base / 'R-CRACKING_CARGA_CRACKING_202503_202508.csv'))

    test_operational['hour'] = test_operational['sampled_date'].dt.floor('1H')
    numeric_cols = test_operational.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
        test_operational_hourly = test_operational.groupby('hour').agg(agg_dict).reset_index()
        new_cols = ['sampled_date']
        for col in numeric_cols:
            new_cols.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max'])
        test_operational_hourly.columns = new_cols

        test_df = pd.merge_asof(test_timestamps.sort_values('sampled_date'), test_operational_hourly.sort_values('sampled_date'), on='sampled_date', direction='nearest', tolerance=pd.Timedelta('30min'))
        test_df = pd.merge_asof(test_df.sort_values('sampled_date'), test_feedstock.sort_values('sampled_date'), on='sampled_date', direction='nearest', tolerance=pd.Timedelta('6H'))
    else:
        test_df = test_timestamps.copy()

    train_features = [col for col in train_df.columns if col not in ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']]
    test_features = [col for col in test_df.columns if col not in ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']]
    common_features = list(set(train_features) & set(test_features))

    train_df = train_df[['sampled_date', 'PCI', 'H2', 'sample_weight'] + common_features].copy()
    test_df = test_df[['sampled_date'] + common_features].copy()
    return train_df, test_df


def prepare_ccr_data(base_path: str = 'data/CCR - Reforming Catalítico'):
    base = Path(base_path)
    train_gas = load_gas_composition(str(base / 'R-RFM_OCT_2209F__202406_202502.csv'))
    train_operational = load_operational_data(str(base / 'Predictoras_202406_202502_CCR.csv'))
    train_bottoms = load_feedstock_properties(str(base / 'r-rfm_oct_FONDO_2102E_202406_202502.csv'))
    train_df = merge_and_aggregate(train_gas, train_operational, train_bottoms)

    test_timestamps = pd.read_csv(base / 'R-RFM_OCT_2209F_202503_202508 - a estimar.csv', sep=';')
    test_timestamps['sampled_date'] = pd.to_datetime(test_timestamps['sampled_date'], errors='coerce')
    test_operational = load_operational_data(str(base / 'Predictoras_202503_202508_CCR.csv'))
    test_bottoms = load_feedstock_properties(str(base / 'r-rfm_oct_FONDO_2102E_202503_202508.csv'))

    test_operational['hour'] = test_operational['sampled_date'].dt.floor('1H')
    numeric_cols = test_operational.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
        test_operational_hourly = test_operational.groupby('hour').agg(agg_dict).reset_index()
        new_cols = ['sampled_date']
        for col in numeric_cols:
            new_cols.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max'])
        test_operational_hourly.columns = new_cols

        test_df = pd.merge_asof(test_timestamps.sort_values('sampled_date'), test_operational_hourly.sort_values('sampled_date'), on='sampled_date', direction='nearest', tolerance=pd.Timedelta('30min'))
        test_df = pd.merge_asof(test_df.sort_values('sampled_date'), test_bottoms.sort_values('sampled_date'), on='sampled_date', direction='nearest', tolerance=pd.Timedelta('6H'))
    else:
        test_df = test_timestamps.copy()

    train_features = [col for col in train_df.columns if col not in ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']]
    test_features = [col for col in test_df.columns if col not in ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']]
    common_features = list(set(train_features) & set(test_features))

    train_df = train_df[['sampled_date', 'PCI', 'H2', 'sample_weight'] + common_features].copy()
    test_df = test_df[['sampled_date'] + common_features].copy()
    return train_df, test_df


def create_basic_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create a small set of physics-informed features for the API (non-exhaustive)"""
    df = X.copy()
    # Temperature related
    temp_cols = [c for c in df.columns if 'temp' in c.lower() or 'temper' in c.lower()]
    flow_cols = [c for c in df.columns if 'flow' in c.lower() or 'caudal' in c.lower() or 'flujo' in c.lower()]

    if temp_cols and flow_cols:
        df['temp_flow_interaction'] = df[temp_cols[0]].fillna(0) * df[flow_cols[0]].fillna(0)

    # Simple ratio of top 3 flows
    if len(flow_cols) >= 2:
        df['flow_ratio'] = df[flow_cols[0]].fillna(0) / (df[flow_cols[1]].fillna(1))

    return df


def add_feature_defaults(X: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Add missing feature columns with default 0. Used at prediction time to align test to training features."""
    df = X.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df
