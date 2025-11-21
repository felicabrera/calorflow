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
            
            # Cargar datos
            train_file = self.data_dir / f"{process.lower()}_train.csv"
            df = pd.read_csv(train_file)
            
            # Preparar datos
            exclude_cols = ['sampled_date', 'PCI', 'H2', 'sample_weight', 'has_actual_measurement']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols]
            y_pci = df['PCI']
            y_h2 = df['H2']
            
            # Configurar entrenamiento
            trainer_config = TrainerConfig()
            if config:
                if 'n_trials' in config:
                    trainer_config.n_trials = config['n_trials']
                if 'cv_folds' in config:
                    trainer_config.cv_folds = config['cv_folds']
                if 'use_autogluon' in config:
                    trainer_config.use_autogluon = config['use_autogluon']
            
            # Entrenar
            trainer = MLTrainer(trainer_config)
            results = trainer.train(X, y_pci, y_h2, process_name=process, output_dir=str(self.models_dir / process))
            
            # Guardar
            trainer.save(str(self.models_dir / process), process)
            
            # Actualizar estado
            self.training_tasks[task_id]["status"] = "completed"
            self.training_tasks[task_id]["results"] = results
            self.training_tasks[task_id]["progress"] = 100
            
        except Exception as e:
            self.training_tasks[task_id]["status"] = "failed"
            self.training_tasks[task_id]["error"] = str(e)
    
    async def get_training_status(self, task_id: str) -> dict:
        """Obtener estado del entrenamiento"""
        if task_id not in self.training_tasks:
            raise Exception(f"Task {task_id} not found")
        
        return self.training_tasks[task_id]
