from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TrainRequest(BaseModel):
    process: str = Field(..., description="FCC or CCR")
    n_trials: Optional[int] = Field(20, description="Number of Optuna trials (default: 20 for API)")
    use_optuna: Optional[bool] = Field(True, description="Whether to run Optuna hyperparameter search")


class PredictRequest(BaseModel):
    process: str = Field(..., description="FCC or CCR")
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries to predict on")


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]


class TrainStatus(BaseModel):
    process: str
    running: bool
    progress: float = 0.0
    last_update: str = ""
    metrics: Optional[Dict[str, Any]] = None
