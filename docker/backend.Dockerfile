FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .


# Install dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy source code
COPY src ./src
COPY data ./data

# Most important part — copy trained model files
COPY artifacts /app/artifacts

# Environment variable for backend to locate artifacts
ENV ARTIFACTS_PATH=/app/artifacts

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]