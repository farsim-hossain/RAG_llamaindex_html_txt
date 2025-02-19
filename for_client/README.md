# Enhanced RAG System with Advanced Context Processing

This is a sophisticated Retrieval-Augmented Generation (RAG) system that combines vector search, document reranking, and advanced context processing to provide accurate answers from a knowledge base.

## Key Features

1. **Advanced Document Processing**

   - Intelligent chunk sizing based on content type
   - Metadata enrichment for better context preservation
   - Multi-language support (German/English)
   - Custom document type classification (FAQ/General)

2. **Enhanced Retrieval Pipeline**

   - Two-stage retrieval process
   - Initial broad search with top-k similarity
   - Cross-encoder reranking for precision
   - Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' for reranking

3. **Optimized Response Generation**

   - Context-aware answer generation
   - Temperature control for consistent outputs
   - Source tracking and attribution
   - Configurable response length

4. **FastAPI Backend**
   - RESTful API interface
   - Async request handling
   - Comprehensive error handling
   - Structured response format

## Technical Stack

- LlamaIndex for vector storage and retrieval
- OpenAI GPT models for text generation
- SentenceTransformers for document reranking
- FastAPI for API framework
- Docker for containerization

## Local Setup

1. Create a `.env` file with:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## Deployment Without Docker

- create virtual environment and activate it using `python -m venv venv` and `source venv/bin/activate`
- pip install -r requirements.txt
- python vectorize_webpages.py
- python rest_api.py or uvicorn rest_api:app --reload

## Deployment with Docker

- docker build -t rag-system .
- docker run -p 8000:8000 --env-file .env rag-system

## API Usage

Query the system:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here"}'
```

If you are using windows cmd, you have to format the command like this :

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"Your question here\"}"
```

Response format:

```json
{
  "answer": "Generated answer",
  "sources": ["source1.txt", "source2.txt"]
}
```
