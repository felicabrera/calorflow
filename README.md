# Calorflow

This repository contains a lightweight API server and a simple web UI that implements the preprocessing and model training logic extracted from the `train_competition.ipynb` notebook. The backend is a FastAPI application with endpoints for preprocessing, model training, and predictions. The UI is a small Bootstrap single page that talks to the API.

## Installation

Create a virtual environment and install dependencies:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the backend

```pwsh
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Open the UI at `http://localhost:8000/` or the FastAPI docs at `http://localhost:8000/docs`.

## Key endpoints

- POST /api/preprocess - Run preprocessing using raw CSV files in `data2/`.
- POST /api/preprocess - Run preprocessing using raw CSV files in `data/` (the repo structure uses `data/`).
- POST /api/train - Start training for process (FCC/CCR).
- GET /api/train/status?process=FCC - Poll training status.
- GET /api/models - List saved models.
- POST /api/predict - Send JSON features to get predictions.

## Frontend

Open `web/index.html` to see the UI. It uses Chart.js for simple metrics graphs.

## Model files and checkpoints

- Models are saved into `models/` as `fcc_pci_xgboost.joblib` etc.
- Training checkpoints `/checkpoints/<process>_results.json` contain the train results.

## Notes

- The training code is a simplified, API-friendly subset of the notebook pipeline. It includes essential preprocessing, baseline ensemble training, and model saving.
- For the full dataset, longer training with Optuna and AutoGluon is recommended — the API supports `n_trials` parameter but the server defaults to a very conservative number (20) to keep responsive in an API environment.
For the full dataset, longer training with Optuna and AutoGluon is recommended — the API supports `n_trials` parameter but the server defaults to a very conservative number (20) to keep responsive in an API environment. Adjust using the UI or call `/api/train` with `n_trials`.

## Docker Compose (all services)

This repository includes a `docker-compose.yml` which creates the following services:
- postgres: Postgres DB for training metadata
- redis: Redis for Celery broker and pub/sub
- app: the FastAPI server
- celery: the Celery worker for background training
- mlflow: minimal MLflow server for tracking

Startup all services with:

```pwsh
docker-compose up -d --build
```

Then visit `http://localhost:8000` and use the UI or `http://localhost:8000/docs` for API docs.
