"""
API Endpoints - Conecta con los módulos existentes en src/
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import sys

# Importar módulos existentes desde src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.trainer import MLTrainer, TrainerConfig
from src.predictor import load_trained_models, predict_from_features
from src.data_utils import load_data
from src.features import create_features

from ..models.schemas import (
    PredictionRequest, PredictionResponse,
    TrainingRequest, TrainingResponse,
    MetricsResponse, VisualizationDataResponse,
    ModelInfo
)
from ..services.predictor import PredictorService
from ..services.trainer import TrainerService
from ..services.visualizations import VisualizationService

router = APIRouter()

# Servicios
predictor_service = PredictorService()
trainer_service = TrainerService()
viz_service = VisualizationService()

# ============================================================================
# MODELOS - Información sobre modelos disponibles
# ============================================================================

@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Lista todos los modelos disponibles"""
    try:
        models = []
        models_dir = Path("models")
        
        for process in ["FCC", "CCR"]:
            process_dir = models_dir / process
            if process_dir.exists():
                for target in ["pci", "h2"]:
                    ensemble_file = process_dir / f"{process}_ensemble_{target}.joblib"
                    if ensemble_file.exists():
                        models.append(ModelInfo(
                            process=process,
                            target=target.upper(),
                            model_type="ensemble",
                            path=str(ensemble_file),
                            available=True
                        ))
        
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# ============================================================================
# PREDICCIONES
# ============================================================================

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Realizar predicción de PCI y H2
    
    Args:
        request: Datos de entrada con features
    
    Returns:
        Predicciones de PCI y H2
    """
    try:
        # Convertir features a DataFrame
        df = pd.DataFrame(request.features, columns=request.feature_names)
        
        # Realizar predicción usando el servicio
        result = await predictor_service.predict(
            df=df,
            process=request.process
        )
        
        return PredictionResponse(
            success=True,
            process=request.process,
            predictions_pci=result["pci"].tolist(),
            predictions_h2=result["h2"].tolist(),
            n_predictions=len(result["pci"])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...), process: str = "FCC"):
    """
    Predicción desde archivo CSV
    
    Args:
        file: Archivo CSV con datos operacionales
        process: Proceso (FCC o CCR)
    
    Returns:
        Predicciones en formato JSON
    """
    try:
        # Leer CSV
        df = pd.read_csv(file.file)
        
        # Aplicar feature engineering
        df = create_features(df)
        
        # Realizar predicción
        result = await predictor_service.predict(df=df, process=process)
        
        # Agregar predicciones al DataFrame
        df['PCI_pred'] = result["pci"]
        df['H2_pred'] = result["h2"]
        
        return {
            "success": True,
            "process": process,
            "n_predictions": len(df),
            "predictions": df[['PCI_pred', 'H2_pred']].to_dict(orient='records'),
            "statistics": {
                "pci": {
                    "mean": float(result["pci"].mean()),
                    "min": float(result["pci"].min()),
                    "max": float(result["pci"].max()),
                    "std": float(result["pci"].std())
                },
                "h2": {
                    "mean": float(result["h2"].mean()),
                    "min": float(result["h2"].min()),
                    "max": float(result["h2"].max()),
                    "std": float(result["h2"].std())
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction error: {str(e)}")

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Entrenar modelo (proceso en background)
    
    Args:
        request: Configuración de entrenamiento
        background_tasks: Tareas en segundo plano
    
    Returns:
        Estado del entrenamiento
    """
    try:
        # Iniciar entrenamiento en background
        task_id = await trainer_service.start_training(
            process=request.process,
            config=request.config,
            background_tasks=background_tasks
        )
        
        return TrainingResponse(
            success=True,
            message=f"Training started for {request.process}",
            task_id=task_id,
            status="training"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.get("/train/status/{task_id}")
async def get_training_status(task_id: str):
    """Obtener estado del entrenamiento"""
    try:
        status = await trainer_service.get_training_status(task_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {str(e)}")

# ============================================================================
# MÉTRICAS Y RESULTADOS
# ============================================================================

@router.get("/metrics/{process}", response_model=MetricsResponse)
async def get_metrics(process: str):
    """
    Obtener métricas de entrenamiento de un proceso
    
    Args:
        process: FCC o CCR
    
    Returns:
        Métricas del modelo
    """
    try:
        metrics_file = Path(f"models/{process}/{process}_training_history.json")
        
        if not metrics_file.exists():
            raise HTTPException(status_code=404, detail=f"No metrics found for {process}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return MetricsResponse(
            success=True,
            process=process,
            metrics=metrics
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")

# ============================================================================
# VISUALIZACIONES - Para mostrar gráficas del notebook en el frontend
# ============================================================================

@router.get("/visualizations/{process}/training", response_model=VisualizationDataResponse)
async def get_training_visualizations(process: str):
    """
    Obtener datos para visualizaciones de entrenamiento
    
    Args:
        process: FCC o CCR
    
    Returns:
        Datos para gráficas (distribuciones, métricas, etc.)
    """
    try:
        viz_data = await viz_service.get_training_visualizations(process)
        return VisualizationDataResponse(
            success=True,
            process=process,
            data=viz_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@router.get("/visualizations/{process}/predictions")
async def get_prediction_visualizations(process: str, limit: int = 100):
    """
    Obtener datos de predicciones para visualizar
    
    Args:
        process: FCC o CCR
        limit: Número máximo de predicciones a retornar
    
    Returns:
        Datos de predicciones recientes
    """
    try:
        viz_data = await viz_service.get_prediction_visualizations(process, limit)
        return {
            "success": True,
            "process": process,
            "data": viz_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@router.get("/visualizations/data-quality/{process}")
async def get_data_quality_visualizations(process: str):
    """
    Obtener datos de calidad de datos para visualizar
    
    Args:
        process: FCC o CCR
    
    Returns:
        Estadísticas de calidad de datos
    """
    try:
        data_file = Path(f"data/processed/{process.lower()}_train.csv")
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail=f"Data file not found for {process}")
        
        df = pd.read_csv(data_file)
        
        quality_data = {
            "total_samples": len(df),
            "total_features": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "target_distributions": {
                "pci": {
                    "mean": float(df['PCI'].mean()) if 'PCI' in df.columns else None,
                    "std": float(df['PCI'].std()) if 'PCI' in df.columns else None,
                    "min": float(df['PCI'].min()) if 'PCI' in df.columns else None,
                    "max": float(df['PCI'].max()) if 'PCI' in df.columns else None,
                    "values": df['PCI'].tolist() if 'PCI' in df.columns else []
                },
                "h2": {
                    "mean": float(df['H2'].mean()) if 'H2' in df.columns else None,
                    "std": float(df['H2'].std()) if 'H2' in df.columns else None,
                    "min": float(df['H2'].min()) if 'H2' in df.columns else None,
                    "max": float(df['H2'].max()) if 'H2' in df.columns else None,
                    "values": df['H2'].tolist() if 'H2' in df.columns else []
                }
            }
        }
        
        return {
            "success": True,
            "process": process,
            "data": quality_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data quality error: {str(e)}")
