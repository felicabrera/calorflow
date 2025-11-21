"""
Script de Entrenamiento - ANCAP DataChallenge 2025
Entrena modelos FCC y CCR usando el módulo MLTrainer
"""

import pandas as pd
from pathlib import Path
from src.trainer import MLTrainer, TrainerConfig

def main():
    print("="*80)
    print("ENTRENAMIENTO DE MODELOS - ANCAP DataChallenge 2025")
    print("="*80)
    
    # Rutas de datos
    data_dir = Path('data/processed')
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Configuración del entrenamiento
    config = TrainerConfig()
    config.n_trials = 100  # Número de trials de Optuna (ajustar según tiempo disponible)
    config.cv_folds = 5
    config.use_autogluon = False  # Desactivar para entrenamiento más rápido
    
    print(f"\nConfiguración:")
    print(f"  - Trials de optimización: {config.n_trials}")
    print(f"  - CV folds: {config.cv_folds}")
    print(f"  - AutoGluon: {config.use_autogluon}")
    
    # ========================================================================
    # ENTRENAR FCC
    # ========================================================================
    print("\n" + "="*80)
    print("1. ENTRENANDO MODELOS FCC")
    print("="*80)
    
    fcc_train = pd.read_csv(data_dir / 'fcc_train.csv')
    print(f"Datos cargados: {len(fcc_train)} muestras, {len(fcc_train.columns)} columnas")
    
    # Preparar features y targets
    exclude_cols = ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']
    feature_cols = [col for col in fcc_train.columns if col not in exclude_cols]
    
    X_fcc = fcc_train[feature_cols]
    y_pci_fcc = fcc_train['PCI']
    y_h2_fcc = fcc_train['H2']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target PCI - rango: [{y_pci_fcc.min():.1f}, {y_pci_fcc.max():.1f}]")
    print(f"Target H2  - rango: [{y_h2_fcc.min():.2f}, {y_h2_fcc.max():.2f}]")
    
    # Entrenar
    trainer_fcc = MLTrainer(config)
    results_fcc = trainer_fcc.train(X_fcc, y_pci_fcc, y_h2_fcc, 
                                     process_name='FCC', 
                                     output_dir=str(models_dir / 'FCC'))
    
    # Guardar modelos
    trainer_fcc.save(str(models_dir / 'FCC'), 'FCC')
    
    print(f"\n✅ FCC completado - RMSE promedio: {results_fcc['competition_score']['rmse_prom']:.2f}")
    
    # ========================================================================
    # ENTRENAR CCR
    # ========================================================================
    print("\n" + "="*80)
    print("2. ENTRENANDO MODELOS CCR")
    print("="*80)
    
    ccr_train = pd.read_csv(data_dir / 'ccr_train.csv')
    print(f"Datos cargados: {len(ccr_train)} muestras, {len(ccr_train.columns)} columnas")
    
    # Preparar features y targets
    feature_cols_ccr = [col for col in ccr_train.columns if col not in exclude_cols]
    
    X_ccr = ccr_train[feature_cols_ccr]
    y_pci_ccr = ccr_train['PCI']
    y_h2_ccr = ccr_train['H2']
    
    print(f"Features: {len(feature_cols_ccr)}")
    print(f"Target PCI - rango: [{y_pci_ccr.min():.1f}, {y_pci_ccr.max():.1f}]")
    print(f"Target H2  - rango: [{y_h2_ccr.min():.2f}, {y_h2_ccr.max():.2f}]")
    
    # Entrenar
    trainer_ccr = MLTrainer(config)
    results_ccr = trainer_ccr.train(X_ccr, y_pci_ccr, y_h2_ccr, 
                                     process_name='CCR',
                                     output_dir=str(models_dir / 'CCR'))
    
    # Guardar modelos
    trainer_ccr.save(str(models_dir / 'CCR'), 'CCR')
    
    print(f"\n✅ CCR completado - RMSE promedio: {results_ccr['competition_score']['rmse_prom']:.2f}")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*80)
    
    print("\nFCC:")
    print(f"  PCI  - RMSE: {results_fcc['metrics_pci']['PCI_rmse']:.2f}, R²: {results_fcc['metrics_pci']['PCI_r2']:.3f}")
    print(f"  H2   - RMSE: {results_fcc['metrics_h2']['H2_rmse']:.2f}, R²: {results_fcc['metrics_h2']['H2_r2']:.3f}")
    print(f"  RMSE_prom: {results_fcc['competition_score']['rmse_prom']:.2f}")
    
    print("\nCCR:")
    print(f"  PCI  - RMSE: {results_ccr['metrics_pci']['PCI_rmse']:.2f}, R²: {results_ccr['metrics_pci']['PCI_r2']:.3f}")
    print(f"  H2   - RMSE: {results_ccr['metrics_h2']['H2_rmse']:.2f}, R²: {results_ccr['metrics_h2']['H2_r2']:.3f}")
    print(f"  RMSE_prom: {results_ccr['competition_score']['rmse_prom']:.2f}")
    
    print(f"\nModelos guardados en: {models_dir.absolute()}")
    print("="*80)


if __name__ == '__main__':
    main()
