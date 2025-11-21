"""
Servicio de Entrenamiento - Usa src/trainer.py
"""

import pandas as pd
from pathlib import Path
import sys
import uuid
from typing import Optional
from fastapi import BackgroundTasks

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.trainer import MLTrainer, TrainerConfig

class TrainerService:
    """Servicio para entrenar modelos"""
    
    def __init__(self):
        self.training_tasks = {}
        self.data_dir = Path("data/processed")
        self.models_dir = Path("models")
    
    async def start_training(self, process: str, config: Optional[dict], background_tasks: BackgroundTasks) -> str:
        """
        Iniciar entrenamiento en background
        
        Args:
            process: FCC o CCR
            config: Configuración de entrenamiento
            background_tasks: FastAPI background tasks
        
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        # Registrar tarea
        self.training_tasks[task_id] = {
            "status": "queued",
            "process": process,
            "progress": 0
        }
        
        # Agregar a background tasks
        background_tasks.add_task(self._train_model, task_id, process, config)
        
        return task_id
    
    async def _train_model(self, task_id: str, process: str, config: Optional[dict]):
        """Función de entrenamiento en background"""
        try:
            # Actualizar estado
            self.training_tasks[task_id]["status"] = "training"
            self.training_tasks[task_id]["progress"] = 10
            self.training_tasks[task_id]["current_step"] = "Cargando datos..."
            
            # Cargar datos
            train_file = self.data_dir / f"{process.lower()}_train.csv"
            df = pd.read_csv(train_file)
            
            self.training_tasks[task_id]["progress"] = 20
            self.training_tasks[task_id]["current_step"] = "Preparando features..."
            
            # Preparar datos
            exclude_cols = ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y_pci = df['PCI']
            y_h2 = df['H2']
            
            self.training_tasks[task_id]["progress"] = 30
            self.training_tasks[task_id]["current_step"] = "Configurando entrenamiento..."
            
            # Configurar entrenamiento
            trainer_config = TrainerConfig()
            if config:
                if 'n_trials' in config:
                    trainer_config.n_trials = config['n_trials']
                    # Si hay pocos trials, entrenar solo los mejores modelos
                    if config['n_trials'] <= 20:
                        trainer_config.models_to_train = ['xgboost', 'lightgbm']
                if 'cv_folds' in config:
                    trainer_config.cv_folds = config['cv_folds']
                if 'use_autogluon' in config:
                    trainer_config.use_autogluon = config['use_autogluon']
            
            self.training_tasks[task_id]["progress"] = 40
            n_models = len(trainer_config.models_to_train)
            self.training_tasks[task_id]["current_step"] = f"Entrenando {n_models} modelos con {trainer_config.n_trials} trials cada uno..."
            
            # Entrenar
            trainer = MLTrainer(trainer_config)
            results = trainer.train(X, y_pci, y_h2, process_name=process, output_dir=str(self.models_dir / process))
            
            self.training_tasks[task_id]["progress"] = 90
            self.training_tasks[task_id]["current_step"] = "Guardando modelos..."
            
            # Guardar
            trainer.save(str(self.models_dir / process), process)
            
            # Actualizar estado con resultados
            self.training_tasks[task_id]["status"] = "completed"
            self.training_tasks[task_id]["progress"] = 100
            self.training_tasks[task_id]["current_step"] = "Completado"
            self.training_tasks[task_id]["results"] = {
                "pci_metrics": {
                    "rmse": float(results.get("pci_rmse", 0)),
                    "mae": float(results.get("pci_mae", 0)),
                    "r2": float(results.get("pci_r2", 0))
                },
                "h2_metrics": {
                    "rmse": float(results.get("h2_rmse", 0)),
                    "mae": float(results.get("h2_mae", 0)),
                    "r2": float(results.get("h2_r2", 0))
                },
                "config": {
                    "n_trials": trainer_config.n_trials,
                    "cv_folds": trainer_config.cv_folds
                }
            }
            
        except Exception as e:
            self.training_tasks[task_id]["status"] = "failed"
            self.training_tasks[task_id]["error"] = str(e)
            self.training_tasks[task_id]["current_step"] = f"Error: {str(e)}"
    
    async def get_training_status(self, task_id: str) -> dict:
        """Obtener estado del entrenamiento"""
        if task_id not in self.training_tasks:
            raise Exception(f"Task {task_id} not found")
        
        return self.training_tasks[task_id]
