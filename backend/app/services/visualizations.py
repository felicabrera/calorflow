"""
Servicio de Visualizaciones - Genera datos para gráficas del frontend
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any

class VisualizationService:
    """Servicio para generar datos de visualización"""
    
    def __init__(self):
        self.data_dir = Path("data/processed")
        self.models_dir = Path("models")
    
    async def get_training_visualizations(self, process: str) -> Dict[str, Any]:
        """
        Obtener datos para visualizaciones de entrenamiento
        
        Genera datos similares a las gráficas del notebook:
        - Distribuciones de PCI y H2
        - Métricas de entrenamiento
        - Feature importance
        """
        try:
            # Cargar datos de entrenamiento
            train_file = self.data_dir / f"{process.lower()}_train.csv"
            df = pd.read_csv(train_file)
            
            # Cargar métricas si existen
            metrics_file = self.models_dir / process / f"{process}_training_history.json"
            metrics = {}
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            
            viz_data = {
                # Distribuciones de targets
                "target_distributions": {
                    "pci": {
                        "values": df['PCI'].dropna().tolist() if 'PCI' in df.columns else [],
                        "mean": float(df['PCI'].mean()) if 'PCI' in df.columns else None,
                        "median": float(df['PCI'].median()) if 'PCI' in df.columns else None,
                        "std": float(df['PCI'].std()) if 'PCI' in df.columns else None,
                        "min": float(df['PCI'].min()) if 'PCI' in df.columns else None,
                        "max": float(df['PCI'].max()) if 'PCI' in df.columns else None
                    },
                    "h2": {
                        "values": df['H2'].dropna().tolist() if 'H2' in df.columns else [],
                        "mean": float(df['H2'].mean()) if 'H2' in df.columns else None,
                        "median": float(df['H2'].median()) if 'H2' in df.columns else None,
                        "std": float(df['H2'].std()) if 'H2' in df.columns else None,
                        "min": float(df['H2'].min()) if 'H2' in df.columns else None,
                        "max": float(df['H2'].max()) if 'H2' in df.columns else None
                    }
                },
                
                # Series temporales (si hay fecha)
                "time_series": None,
                
                # Métricas de entrenamiento
                "metrics": metrics,
                
                # Estadísticas del dataset
                "dataset_stats": {
                    "n_samples": len(df),
                    "n_features": len(df.columns) - 2,  # Excluir PCI y H2
                    "missing_values": int(df.isnull().sum().sum())
                }
            }
            
            # Agregar series temporales si hay columna de fecha
            if 'sampled_date' in df.columns:
                df['sampled_date'] = pd.to_datetime(df['sampled_date'])
                df_sorted = df.sort_values('sampled_date')
                
                viz_data["time_series"] = {
                    "dates": df_sorted['sampled_date'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    "pci": df_sorted['PCI'].tolist() if 'PCI' in df_sorted.columns else [],
                    "h2": df_sorted['H2'].tolist() if 'H2' in df_sorted.columns else []
                }
            
            return viz_data
            
        except Exception as e:
            raise Exception(f"Error generating visualization data: {str(e)}")
    
    async def get_prediction_visualizations(self, process: str, limit: int = 100) -> Dict[str, Any]:
        """
        Obtener datos de predicciones para visualizar
        """
        # Por ahora retornar datos de ejemplo
        # En producción, esto vendría de una BD con predicciones guardadas
        return {
            "message": "Prediction visualizations not yet implemented",
            "process": process
        }
