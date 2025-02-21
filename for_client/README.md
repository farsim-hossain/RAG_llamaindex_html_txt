# Enhanced RAG System with Advanced Context Processing

This is a sophisticated **Retrieval-Augmented Generation (RAG)** system that integrates vector search, document reranking, and advanced context processing to provide **highly accurate** answers from a knowledge base.

## 🚀 Key Features

### 🔍 Advanced Document Processing

- **Intelligent chunk sizing** based on content type
- **Metadata enrichment** for better context preservation
- **Multi-language support** (German/English)
- **Custom document type classification** (FAQ/General)

### 🎯 Enhanced Retrieval Pipeline

- **Two-stage retrieval process**
- **Initial broad search** with top-k similarity
- **Cross-encoder reranking** for precision
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking

### 🧠 Optimized Response Generation

- **Context-aware** answer generation
- **Temperature control** for consistent outputs
- **Source tracking** and attribution
- **Configurable response length**

### ⚡ FastAPI Backend

- **RESTful API** interface
- **Async request handling**
- **Comprehensive error handling**
- **Structured response format**

### 📦 Containerized Deployment

- **Simplified deployment** using `docker-compose`
- **Pre-built Nginx reverse proxy** for load balancing & routing
- **Scalable architecture** with multiple RAG service replicas

## 🛠 Technical Stack

- **LlamaIndex** for vector storage and retrieval
- **OpenAI GPT models** for text generation
- **SentenceTransformers** for document reranking
- **FastAPI** for API framework
- **Docker** for containerization
- **Nginx** as a reverse proxy (via Docker)

---

## 🏗 Local Setup

### 🔧 Prerequisites

Ensure **Docker** and **Docker Compose** are installed on your machine:

```sh
docker --version
docker-compose --version
```

### Docker and Docker Compose Installation

Please follow this link for installing Docker :
https://docs.docker.com/engine/install/rhel/#install-using-the-repository

Please follow this link for installing Docker Compose :
https://docs.docker.com/compose/install/linux/

# Create a .env File

Create a `.env` file in the root directory (if not there already) of the project with the following content:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

# Prepare the Vector Store

If the `vector_store` directory does not exist, you need to generate it. Follow these steps:

## Create a Virtual Environment and Activate It:

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install Depeendencies:

```sh
pip install -r requirements.txt
```

## Generate the Vector Store:

```sh
python vectorize_webpages.py
```

# Deployment with Docker Compose

## Start the Services :

```sh
docker-compose up -d
```

## Verify the Services :

```sh
docker ps
```

## Stop the Services

```sh
docker-compose down
```

# API Usage

Query the system using the following curl command:

```sh
curl -X POST "http://localhost/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here"}'
```

If you are using Windows CMD, format the command like this:

```sh
curl -X POST "http://localhost/query" -H "Content-Type: application/json" -d "{\"question\": \"Your question here\"}"
```

## Response Format

The API will return a JSON response in the following format:

```json
{
  "answer": "Generated answer",
  "sources": ["source1.txt", "source2.txt"]
}
```

# Additional Notes

## Nginx Configuration :

The Nginx configuration is defined in the nginx.conf file. If you need to customize the routing or add SSL support, modify this file and restart the services:

```sh
docker-compose up -d
```

## Scaling the RAG Service :

You can scale the rag-api service to handle more traffic by specifying the number of replicas in the docker-compose.yml file or using the following command:

```sh
docker-compose up -d --scale rag-api=3
```

## Health Checks :

The docker-compose.yml file includes a health check for the rag-api service. Ensure your FastAPI application has a /health endpoint defined:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```
