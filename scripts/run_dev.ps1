# Run uvicorn development server for FastAPI
$env:PYTHONPATH = '.'
uvicorn src.app:app --reload --port 8000 --host 0.0.0.0
