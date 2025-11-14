# Calorflow

Sistema de predicción de PCI y H2 para procesos de refinería (FCC/CCR) con API REST completa.

## Estructura del Proyecto

```
calorflow/
├── src/
│   ├── api/
│   │   ├── main.py              # Aplicación FastAPI principal
│   │   ├── api_endpoints.py     # Implementación de endpoints
│   │   ├── api_schemas.py       # Schemas Pydantic
│   │   ├── api_helpers.py       # Funciones auxiliares
│   │   └── logging_config.py    # Configuración de logs
│   ├── data_utils.py            # Utilidades de datos
│   ├── features.py              # Feature engineering
│   ├── trainer.py               # Entrenamiento de modelos
│   └── predictor.py             # Predicciones
├── data/                        # Datasets
├── models/                      # Modelos entrenados
├── predictions/                 # Predicciones generadas
├── logs/                        # Logs de la API
├── run_api.py                   # Script para ejecutar la API
├── test_api.py                  # Tests de la API
└── requirements.txt             # Dependencias

```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso del Backend

### Iniciar la API

```bash
python scripts/run_api.py
```

La API estará disponible en:
- API: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs
- Documentación alternativa: http://localhost:8000/redoc

### Endpoints Disponibles

#### General
- `GET /` - Información general de la API
- `GET /health` - Estado del sistema y modelos disponibles
- `GET /models` - Lista de modelos entrenados
- `GET /process-info/{process_name}` - Información del proceso (FCC/CCR)

#### Entrenamiento
- `POST /train` - Entrenar modelos para FCC o CCR

#### Predicción
- `POST /predict` - Generar predicciones
- `POST /batch-predict` - Predicciones en lote
- `GET /download-prediction/{filename}` - Descargar predicciones

#### Competencia
- `POST /submission` - Generar archivo de submission

#### Datos
- `POST /data-quality` - Verificar calidad de datos
- `POST /upload-data` - Subir archivos CSV

### Ejemplos de Uso

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

#### 2. Entrenar Modelo

```python
import requests

payload = {
    "process_name": "FCC",
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv",
    "output_dir": "models"
}

response = requests.post("http://localhost:8000/train", json=payload)
print(response.json())
```

#### 3. Generar Predicciones

```python
import requests

payload = {
    "process_name": "FCC",
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202503_202508_FCC.csv",
    "model_dir": "models"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

#### 4. Generar Submission

```python
import requests

payload = {
    "process_name": "FCC",
    "model_dir": "models",
    "data_dir": "data",
    "output_dir": "predictions"
}

response = requests.post("http://localhost:8000/submission", json=payload)
print(response.json())
```

#### 5. Verificar Calidad de Datos

```python
import requests

payload = {
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv",
    "target_cols": ["PCI", "H2"]
}

response = requests.post("http://localhost:8000/data-quality", json=payload)
print(response.json())
```

### Tests Automatizados

```bash
python scripts/test_api.py
```

## Configuración

Crea un archivo `.env` basado en `.env.example`:

```env
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
LOG_LEVEL=INFO
LOG_DIR=logs
MODEL_DIR=models
DATA_DIR=data
PREDICTIONS_DIR=predictions
CORS_ORIGINS=*
```

## Logs

Los logs se guardan en la carpeta `logs/`:
- `calorflow_YYYYMMDD.log` - Logs generales
- `calorflow_errors_YYYYMMDD.log` - Solo errores

## Documentación Interactiva

FastAPI genera documentación automática:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
