

# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt


# Copy application code
COPY . .

# Create vector store directory
RUN mkdir -p vector_store

# Expose port
EXPOSE 8000

# Run vectorizer first, then start the API
CMD ["sh", "-c", "uvicorn rest_api:app --host 0.0.0.0 --port 8000"]
