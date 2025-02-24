
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from llama_index.core import PromptHelper
# from llama_index.core import VectorStoreIndex
# from llama_index.llms.openai import OpenAI # Updated import
# from llama_index.core import StorageContext, load_index_from_storage
# from sentence_transformers import CrossEncoder
# from llama_index.core.response_synthesizers import get_response_synthesizer
# import os
# from dotenv import load_dotenv
# from contextlib import asynccontextmanager
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# # Define the request body model
# class QueryRequest(BaseModel):
#     question: str

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Enhanced RAG API is starting up with the following components:")
#     print(f"- Index: LlamaIndex")
#     print(f"- LLM: {OPENAI_MODEL}")
#     print(f"- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
#     print(f"- Key features: MMR retrieval, document reranking, key concept extraction, query reformulation")
#     yield
#     print("Shutting down Enhanced RAG API")

# # Initialize FastAPI app with lifespan
# app = FastAPI(
#     lifespan=lifespan,
#     title="Enhanced RAG API",
#     description="Retrieval-Augmented Generation system with advanced context processing"
# )

# # Initialize components
# def initialize_components():
#     vector_store_path = "vector_store"
#     if not os.path.exists(vector_store_path):
#         raise FileNotFoundError(f"Vector store directory not found at: {vector_store_path}")
    
#     # Initialize the LLM
#     llm = OpenAI(
#         api_key=OPENAI_API_KEY,
#         model=OPENAI_MODEL,
#         temperature=0.2,
#         max_tokens=4000
#     )
    
#     # Load the index using storage context
#     storage_context = StorageContext.from_defaults(persist_dir=vector_store_path)
#     index = load_index_from_storage(storage_context)
    
#     # Initialize the reranker
#     reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
#     return {
#         'index': index,
#         'reranker': reranker,
#         'llm': llm
#     }


# # Initialize components
# components = initialize_components()

# def rerank_documents(question, docs, top_k=3):
#     if not docs:
#         logger.info("No documents to rerank.")
#         return []
    
#     pairs = [[question, doc.get_text()] for doc in docs]
#     scores = components['reranker'].predict(pairs)
    
#     scored_docs = list(zip(docs, scores))
#     scored_docs.sort(key=lambda x: x[1], reverse=True)
    
#     logger.info(f"Reranked documents with scores: {[score for _, score in scored_docs]}")
    
#     return [doc for doc, score in scored_docs[:top_k]]

# @app.post("/query", response_model=dict)
# async def query_rag_system(request: QueryRequest):
#     try:
#         # Get reformulated query
#         reformulated_query = request.question
        
#         response_synthesizer = get_response_synthesizer(
#             response_mode="tree_summarize",
#             verbose=True
#         )

#         custom_prompt = """
#         Please provide a comprehensive and detailed answer to the following question. 
#         Include relevant examples, explanations, and context where applicable.

#         Question: {query_str}

#         Please use the following context to formulate your response:
#         {context_str}

#         Detailed answer:
#         """


#         # Get documents using the index
#         query_engine = components['index'].as_query_engine(
#             similarity_top_k=10,
#             llm=components['llm'],
#             context_window=4096,
#             response_synthesizer=response_synthesizer,
#             text_qa_template=custom_prompt
#         )
#         raw_docs = query_engine.retrieve(reformulated_query)
        
#         # Rerank the retrieved documents
#         reranked_docs = rerank_documents(reformulated_query, raw_docs)
#         logger.info(f"Reranked {len(reranked_docs)} documents.")
        
#         # Generate answer using the index
#         query_engine = components['index'].as_query_engine(
#             similarity_top_k=5,
#             llm=components['llm']
#         )
#         response = query_engine.query(reformulated_query)
        
#         # Extract source information using the correct attribute
#         sources = [doc.metadata.get('source_file', 'unknown') for doc in reranked_docs]
        
#         return {
#             "answer": response.response,
#             "sources": sources
#         }
#     except Exception as e:
#         import traceback
#         logger.error(f"Detailed error: {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



###################### ENFORCING GERMAN LANGUAGE WITH AN AGENT ##########################
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import PromptHelper
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from sentence_transformers import CrossEncoder
from llama_index.core.response_synthesizers import get_response_synthesizer
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

class GermanReasoningAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.validation_prompt = """
        Du bist ein Sprachvalidierungs- und Korrekturagent. Deine Aufgabe ist es:
        1. Zu überprüfen, ob der folgende Text auf Deutsch ist
        2. Falls nicht, übersetze ihn ins Deutsche
        3. Optimiere die deutsche Formulierung für Natürlichkeit und Klarheit
        
        Text zur Überprüfung:
        {text}
        
        Bitte gib nur den überprüften/korrigierten deutschen Text zurück, ohne zusätzliche Erklärungen.
        """

    async def ensure_german_response(self, text: str) -> str:
        validation_query = self.validation_prompt.format(text=text)
        
        # Create a structured prompt for the LLM
        messages = [
            {"role": "system", "content": "Du bist ein deutscher Sprachexperte. Antworte ausschließlich auf Deutsch."},
            {"role": "user", "content": validation_query}
        ]
        
        try:
            response = await self.llm.acomplete(messages=messages)
            return response.message.content
        except Exception as e:
            logger.error(f"Error in German validation: {str(e)}")
            return text  # Fallback to original text if validation fails

class QueryRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Enhanced German RAG API is starting up with the following components:")
    print(f"- Index: LlamaIndex")
    print(f"- LLM: {OPENAI_MODEL}")
    print(f"- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"- German Reasoning Agent: Enabled")
    yield
    print("Shutting down Enhanced German RAG API")

app = FastAPI(
    lifespan=lifespan,
    title="Enhanced German RAG API",
    description="Retrieval-Augmented Generation system with German language enforcement"
)

def initialize_components():
    vector_store_path = "vector_store"
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store directory not found at: {vector_store_path}")

    llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=4000
    )

    storage_context = StorageContext.from_defaults(persist_dir=vector_store_path)
    index = load_index_from_storage(storage_context)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    german_agent = GermanReasoningAgent(llm)

    return {
        'index': index,
        'reranker': reranker,
        'llm': llm,
        'german_agent': german_agent
    }

components = initialize_components()

def rerank_documents(question, docs, top_k=3):
    if not docs:
        logger.info("No documents to rerank.")
        return []

    pairs = [[question, doc.get_text()] for doc in docs]
    scores = components['reranker'].predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Reranked documents with scores: {[score for _, score in scored_docs]}")
    return [doc for doc, score in scored_docs[:top_k]]

@app.post("/query", response_model=dict)
async def query_rag_system(request: QueryRequest):
    try:
        # Enhanced German prompt
        reformulated_query = f"""
        Beantworte die folgende Frage ausführlich auf Deutsch. 
        Frage: {request.question}
        """

        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            verbose=True
        )

        custom_prompt = """
        Bitte geben Sie eine umfassende und detaillierte Antwort auf Deutsch.
        
        Frage: {query_str}
        
        Kontext zur Berücksichtigung:
        {context_str}
        
        Wichtige Anweisungen:
        1. Die Antwort MUSS auf Deutsch sein
        2. Verwenden Sie natürliche, flüssige deutsche Sprache
        3. Fachbegriffe sollten korrekt übersetzt werden
        4. Die Erklärung sollte klar und verständlich sein
        
        Ihre detaillierte Antwort auf Deutsch:
        """

        query_engine = components['index'].as_query_engine(
            similarity_top_k=10,
            llm=components['llm'],
            context_window=4096,
            response_synthesizer=response_synthesizer,
            text_qa_template=custom_prompt
        )
        
        raw_docs = query_engine.retrieve(reformulated_query)
        reranked_docs = rerank_documents(reformulated_query, raw_docs)
        
        # Generate initial response
        response = query_engine.query(reformulated_query)
        
        # Ensure response is in German using the reasoning agent
        validated_response = await components['german_agent'].ensure_german_response(response.response)
        
        sources = [doc.metadata.get('source_file', 'unbekannt') for doc in reranked_docs]
        
        return {
            "answer": validated_response,
            "sources": sources
        }

    except Exception as e:
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ein Fehler ist aufgetreten: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)