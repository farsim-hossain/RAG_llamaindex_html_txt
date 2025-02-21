# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code (including existing vector_store)
COPY . .

# Expose port
EXPOSE 8000

# Run the API with multiple workers for better performance
CMD ["uvicorn", "rest_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
