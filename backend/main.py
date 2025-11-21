"""
FastAPI Backend for Calorflow - ANCAP DataChallenge 2025
Servidor API que expone modelos de ML y visualizaciones
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import sys
from pathlib import Path

# Agregar src/ al path para importar módulos existentes
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.api_endpoints import router as api_router
from app.config import settings

# Configuración del ciclo de vida de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    print("🚀 Iniciando Calorflow API...")
    print(f"📊 Modelos disponibles en: {settings.MODELS_DIR}")
    print(f"📁 Datos en: {settings.DATA_DIR}")
    yield
    print("👋 Cerrando Calorflow API...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Calorflow API",
    description="API para predicción de PCI y H2 en procesos FCC y CCR",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS para permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite y React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas de la API
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint con información de la API"""
    return {
        "name": "Calorflow API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "models": "/api/v1/models",
            "predict": "/api/v1/predict",
            "train": "/api/v1/train",
            "metrics": "/api/v1/metrics",
            "visualizations": "/api/v1/visualizations"
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Calorflow API is running"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
