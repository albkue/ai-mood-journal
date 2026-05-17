FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/fastapi_server/requirements.txt .
COPY ml/requirements.txt ./ml_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r ml_requirements.txt

# Download NLTK data during build to avoid runtime delays
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY backend/fastapi_server /app
COPY ml /ml

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
