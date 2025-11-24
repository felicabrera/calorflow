FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install mlflow
ENV MLFLOW_VERSION=2.21.3
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir mlflow==${MLFLOW_VERSION} psycopg2-binary

# Create mount point for mlruns artifacts
RUN mkdir -p /mlflow && mkdir -p /mlflow/artifacts

WORKDIR /app

EXPOSE 5000

ENV MLFLOW_BACKEND_STORE_URI="postgresql://postgres:postgres@postgres:5432/calorflow"
CMD ["sh", "-lc", "mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root /mlflow/artifacts --backend-store-uri ${MLFLOW_BACKEND_STORE_URI}"]
