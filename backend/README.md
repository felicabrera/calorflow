# Calorflow Backend

FastAPI backend service for ML inference and database interaction.

## Structure

- `app/` - Core backend application code
  - `api/` - API route handlers
  - `models/` - Pydantic schemas and data models
  - `services/` - Business logic (ML training, prediction, feature engineering)
  - `utils/` - Helper functions and utilities
  - `main.py` - FastAPI application entry point
  - `config.py` - Configuration settings
  - `logging_config.py` - Logging setup

- `tests/` - Backend unit and integration tests
- `requirements.txt` - Backend dependencies
- `Dockerfile` - Docker container configuration

## Running the API

From the project root:

```bash
python scripts/run_api.py
```

Or directly with uvicorn:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

Build the container:

```bash
docker build -t calorflow-backend ./backend
```

Run the container:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data calorflow-backend
```

## API Documentation

Once running, visit:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
