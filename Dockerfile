FROM python:3.11-slim

WORKDIR /app

# Install basic system deps
RUN apt-get update && apt-get install -y build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy app
COPY . /app

# Create non-root user
RUN useradd --uid 1000 --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
