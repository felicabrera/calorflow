"""
Configuración del Backend
"""

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Directorios del proyecto
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Calorflow API"
    VERSION: str = "1.0.0"
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]
    
    # ML Settings
    DEFAULT_PROCESS: str = "FCC"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    class Config:
        case_sensitive = True

settings = Settings()
