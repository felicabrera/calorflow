"""
Servicio de Predicción - Usa los módulos existentes de src/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Importar desde src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.trainer import MLTrainer
from src.features import create_features

class PredictorService:
    """Servicio para realizar predicciones"""
    
    def __init__(self):
        self.models = {}
        self.models_dir = Path("models")
    
    def load_model(self, process: str):
        """Cargar modelo si no está en cache"""
        if process not in self.models:
            try:
                model_dir = self.models_dir / process
                trainer = MLTrainer.load(str(model_dir), process)
                self.models[process] = trainer
            except Exception as e:
                raise Exception(f"Error loading model for {process}: {str(e)}")
        
        return self.models[process]
    
    async def predict(self, df: pd.DataFrame, process: str) -> dict:
        """
        Realizar predicción
        
        Args:
            df: DataFrame con features
            process: FCC o CCR
        
        Returns:
            Dict con predicciones de PCI y H2
        """
        # Cargar modelo
        trainer = self.load_model(process)
        
        # Aplicar feature engineering
        df_features = create_features(df.copy())
        
        # Realizar predicción
        pci_pred, h2_pred = trainer.predict(df_features)
        
        return {
            "pci": pci_pred,
            "h2": h2_pred
        }
