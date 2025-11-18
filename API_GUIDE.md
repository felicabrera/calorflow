# Calorflow API - Guía de Uso

## Inicio Rápido

### Windows
```bash
start_api.bat
```

### Linux/Mac
```bash
chmod +x start_api.sh
./start_api.sh
```

### Manual
```bash
pip install -r requirements.txt
python run_api.py
```

## Documentación Interactiva

Una vez iniciada la API, abre tu navegador en:
- http://localhost:8000/docs

Aquí podrás:
- Ver todos los endpoints disponibles
- Probar cada endpoint directamente desde el navegador
- Ver los schemas de request/response
- Descargar la especificación OpenAPI

## Ejemplos con cURL

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Listar Modelos Disponibles
```bash
curl http://localhost:8000/models
```

### 3. Información de Proceso
```bash
curl http://localhost:8000/process-info/FCC
curl http://localhost:8000/process-info/CCR
```

### 4. Verificar Calidad de Datos
```bash
curl -X POST "http://localhost:8000/data-quality" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv",
    "target_cols": ["PCI", "H2"]
  }'
```

### 5. Generar Predicciones
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "process_name": "FCC",
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202503_202508_FCC.csv",
    "model_dir": "models"
  }'
```

### 6. Generar Submission para Competencia
```bash
curl -X POST "http://localhost:8000/submission" \
  -H "Content-Type: application/json" \
  -d '{
    "process_name": "FCC",
    "model_dir": "models",
    "data_dir": "data",
    "output_dir": "predictions"
  }'
```

### 7. Entrenar Modelo
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "process_name": "FCC",
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv",
    "output_dir": "models"
  }'
```

### 8. Subir Archivo
```bash
curl -X POST "http://localhost:8000/upload-data" \
  -F "file=@data.csv" \
  -F "process_name=FCC"
```

### 9. Descargar Predicción
```bash
curl -O "http://localhost:8000/download-prediction/FCC_submission_20251114.csv"
```

## Ejemplos con Python

### Instalación del cliente
```bash
pip install requests
```

### Health Check
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

### Predicción Simple
```python
import requests

payload = {
    "process_name": "FCC",
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202503_202508_FCC.csv",
    "model_dir": "models"
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()

print(f"Samples: {result['n_samples']}")
print(f"PCI predictions: {result['predictions_pci'][:5]}...")
print(f"H2 predictions: {result['predictions_h2'][:5]}...")
```

### Predicción en Lote
```python
import requests

payload = {
    "process_name": "FCC",
    "file_paths": [
        "data/FCC - Cracking Catalítico/Predictoras_202503_202508_FCC.csv",
        "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv"
    ],
    "model_dir": "models",
    "output_dir": "predictions"
}

response = requests.post("http://localhost:8000/batch-predict", json=payload)
result = response.json()

print(f"Total files: {result['total_files']}")
print(f"Successful: {result['successful']}")
print(f"Output files: {result['output_files']}")
```

### Entrenar con Configuración Personalizada
```python
import requests

payload = {
    "process_name": "FCC",
    "data_path": "data/FCC - Cracking Catalítico/Predictoras_202406_202502_FCC.csv",
    "config": {
        "random_seed": 42,
        "n_trials": 100,
        "cv_folds": 5,
        "use_time_series_cv": True,
        "models_to_train": ["xgboost", "lightgbm", "catboost"]
    },
    "output_dir": "models"
}

response = requests.post("http://localhost:8000/train", json=payload)
result = response.json()

print(f"Training time: {result['training_time']:.2f}s")
print(f"PCI R²: {result['metrics_pci']['PCI_r2']:.4f}")
print(f"H2 R²: {result['metrics_h2']['H2_r2']:.4f}")
```

## Ejemplos con JavaScript/Fetch

### Health Check
```javascript
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));
```

### Predicción
```javascript
const payload = {
  process_name: "FCC",
  data_path: "data/FCC - Cracking Catalítico/Predictoras_202503_202508_FCC.csv",
  model_dir: "models"
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => {
    console.log(`Samples: ${data.n_samples}`);
    console.log(`Statistics:`, data.statistics);
  });
```

## Schemas de Request/Response

Todos los schemas están documentados en `/docs`. Los principales son:

### TrainingRequest
```json
{
  "process_name": "FCC",
  "data_path": "string",
  "config": {
    "random_seed": 42,
    "n_trials": 300,
    "cv_folds": 5,
    "use_time_series_cv": true,
    "models_to_train": ["xgboost", "lightgbm", "catboost", "randomforest"]
  },
  "output_dir": "models"
}
```

### PredictionRequest
```json
{
  "process_name": "FCC",
  "data_path": "string",
  "model_dir": "models",
  "return_features": false
}
```

### SubmissionRequest
```json
{
  "process_name": "FCC",
  "model_dir": "models",
  "data_dir": "data",
  "output_dir": "predictions"
}
```

## Tests Automatizados

```bash
python test_api.py
```

Este script ejecuta todos los tests de los endpoints principales.

## Configuración

Variables de entorno disponibles (archivo `.env`):

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

Los logs se guardan automáticamente en la carpeta `logs/`:
- `calorflow_YYYYMMDD.log` - Todos los logs
- `calorflow_errors_YYYYMMDD.log` - Solo errores

## Solución de Problemas

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: Port already in use
Cambia el puerto en `.env` o cierra la aplicación que usa el puerto 8000.

### Error: Models not found
Entrena los modelos primero usando el endpoint `/train`.

### Error: Data file not found
Verifica que las rutas a los archivos CSV sean correctas.

## Integración con Frontend

La API soporta CORS por defecto, permitiendo llamadas desde cualquier origen.

### Ejemplo React
```javascript
const [predictions, setPredictions] = useState(null);

const handlePredict = async () => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      process_name: 'FCC',
      data_path: 'data/test.csv',
      model_dir: 'models'
    })
  });
  
  const data = await response.json();
  setPredictions(data);
};
```

## Documentación Adicional

- OpenAPI Spec: http://localhost:8000/openapi.json
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
