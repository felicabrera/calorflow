from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import logging
import time

from .api.api_endpoints import (
    train_model_endpoint,
    predict_endpoint,
    batch_predict_endpoint,
    generate_submission_endpoint,
    check_data_quality_endpoint,
    list_models_endpoint,
    health_check_endpoint
)
from .models.api_schemas import (
    TrainingRequest, TrainingResponse,
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    SubmissionRequest, SubmissionResponse,
    DataQualityRequest, DataQualityResponse,
    ModelListResponse, HealthCheckResponse,
    ErrorResponse, ErrorDetail
)
from .logging_config import setup_logging


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Calorflow API")
    logger.info("Checking system dependencies...")
    
    try:
        health = health_check_endpoint()
        logger.info(f"System status: {health['status']}")
        logger.info(f"Models available: {health['models_available']}")
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
    
    yield
    
    logger.info("Shutting down Calorflow API")


app = FastAPI(
    title="Calorflow API",
    description="API para predicción de PCI y H2 en procesos de refinería (FCC/CCR)",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
        }
    )


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Calorflow API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/models",
            "train": "/train",
            "predict": "/predict",
            "batch-predict": "/batch-predict",
            "submission": "/submission",
            "data-quality": "/data-quality"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health_check(model_dir: str = "models"):
    try:
        result = health_check_endpoint(model_dir)
        return result
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/models", response_model=ModelListResponse, tags=["Models"])
async def list_models(model_dir: str = "models"):
    try:
        result = list_models_endpoint(model_dir)
        return result
    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(request: TrainingRequest):
    logger.info(f"Training request for process: {request.process_name}")
    
    try:
        result = train_model_endpoint(request)
        logger.info(f"Training completed: {result.get('process_name')}")
        return result
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    logger.info(f"Prediction request for process: {request.process_name}")
    
    try:
        result = predict_endpoint(request)
        logger.info(f"Prediction completed: {result.get('n_samples')} samples")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    logger.info(f"Batch prediction request for process: {request.process_name}")
    logger.info(f"Files to process: {len(request.file_paths)}")
    
    try:
        result = batch_predict_endpoint(request)
        logger.info(f"Batch prediction completed: {result.get('successful')}/{result.get('total_files')} successful")
        return result
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/submission", response_model=SubmissionResponse, tags=["Competition"])
async def generate_submission(request: SubmissionRequest):
    logger.info(f"Submission request for process: {request.process_name}")
    
    try:
        result = generate_submission_endpoint(request)
        logger.info(f"Submission generated: {result.get('submission_file')}")
        return result
    except Exception as e:
        logger.error(f"Submission error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/data-quality", response_model=DataQualityResponse, tags=["Data"])
async def check_data_quality(request: DataQualityRequest):
    logger.info("Data quality check request")
    
    try:
        result = check_data_quality_endpoint(request)
        logger.info(f"Data quality score: {result.get('quality_score')}")
        return result
    except Exception as e:
        logger.error(f"Data quality error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/upload-data", tags=["Data"])
async def upload_data(
    file: UploadFile = File(...),
    process_name: str = "FCC",
    destination: str = "data/uploads"
):
    logger.info(f"File upload: {file.filename}")
    
    try:
        upload_dir = Path(destination)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        return {
            "success": True,
            "filename": file.filename,
            "path": str(file_path),
            "size_bytes": len(content),
            "process_name": process_name
        }
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/download-prediction/{filename}", tags=["Prediction"])
async def download_prediction(filename: str, directory: str = "predictions"):
    file_path = Path(directory) / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {filename}"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/csv"
    )


@app.get("/process-info/{process_name}", tags=["General"])
async def get_process_info(process_name: str):
    process_info = {
        "FCC": {
            "name": "Fluid Catalytic Cracking",
            "description": "Proceso de craqueo catalítico fluido",
            "targets": ["PCI", "H2"],
            "typical_range_pci": [8.0, 12.0],
            "typical_range_h2": [0.0, 5.0]
        },
        "CCR": {
            "name": "Continuous Catalytic Reforming",
            "description": "Proceso de reformado catalítico continuo",
            "targets": ["PCI", "H2"],
            "typical_range_pci": [8.0, 12.0],
            "typical_range_h2": [0.0, 15.0]
        }
    }
    
    if process_name not in process_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process not found: {process_name}. Available: FCC, CCR"
        )
    
    return process_info[process_name]


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
