# rest_api.py

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Define the request body model
# class QueryRequest(BaseModel):
#     question: str

# # Initialize FastAPI app
# app = FastAPI()

# # Load the pre-built vector store
# vector_store_path = "vector_store"  # Path to your vector store
# if not os.path.exists(vector_store_path):
#     raise FileNotFoundError(f"Vector store not found at: {vector_store_path}")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

# # Initialize Groq LLM
# llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

# # Initialize memory for chat history
# memory = ConversationBufferMemory()

# # Create a RetrievalQA chain with memory
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(),
#     memory=memory,
#     verbose=True
# )

# @app.post("/query")
# async def query_rag_system(request: QueryRequest):
#     """
#     Endpoint to query the RAG system with memory.
#     """
#     try:
#         # Get the answer from the RetrievalQA chain
#         response = qa_chain.invoke({"query": request.question})
#         return {"answer": response["result"]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# # Run the app with Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


######################## WITH OPEN AI ########################

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
# from langchain_openai import ChatOpenAI  # Import OpenAI's Chat model
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Use OpenAI API key
# OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# # Define the request body model
# class QueryRequest(BaseModel):
#     question: str

# # Initialize FastAPI app
# app = FastAPI()

# # Load the pre-built vector store
# vector_store_path = "vector_store"  # Path to your vector store
# if not os.path.exists(vector_store_path):
#     raise FileNotFoundError(f"Vector store not found at: {vector_store_path}")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

# # Initialize OpenAI LLM
# llm = ChatOpenAI(
#     api_key=OPENAI_API_KEY,
#     model=OPENAI_MODEL,  # Use GPT-3.5 Turbo
#     temperature=0.7,        # Adjust creativity (0.0 = deterministic, 1.0 = creative)
#     max_tokens=2000          # Limit response length
# )

# # Initialize memory for chat history
# memory = ConversationBufferMemory()

# # Create a RetrievalQA chain with memory
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(),
#     memory=memory,
#     verbose=True
# )

# @app.post("/query")
# async def query_rag_system(request: QueryRequest):
#     """
#     Endpoint to query the RAG system with memory.
#     """
#     try:
#         # Get the answer from the RetrievalQA chain
#         response = qa_chain.invoke({"query": request.question})
#         return {"answer": response["result"]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# # Run the app with Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

############################### Enhanced RAG with OpenAI #####################################

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_openai import ChatOpenAI
# from sentence_transformers import CrossEncoder
# from collections import Counter
# import re, os, nltk
# from nltk.corpus import stopwords
# from dotenv import load_dotenv
# from contextlib import asynccontextmanager

# # Download required NLTK data
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)

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
#     print(f"- Vector Store: FAISS")
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
#         raise FileNotFoundError(f"Vector store not found at: {vector_store_path}")
    
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
#     llm = ChatOpenAI(
#         api_key=OPENAI_API_KEY,
#         model=OPENAI_MODEL,
#         temperature=0.2,
#         max_tokens=2000
#     )
    
#     retriever = vector_store.as_retriever(
#         search_type="mmr", 
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 10,
#             "lambda_mult": 0.7
#         }
#     )
    
#     reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
#     memory = ConversationBufferWindowMemory(
#         memory_key="chat_history",
#         input_key="query",
#         output_key="result",
#         k=5
#     )
    
#     custom_prompt = PromptTemplate(
#         template="""
# Answer the question based on the following context:

# Context: {context}

# Question: {query}

# Additional Instructions:
# 1. If the context doesn't provide enough information, say so instead of making up answers.
# 2. Use specific details from the context to support your answer.
# 3. Keep your answer concise but comprehensive.

# Answer:
# """,
#         input_variables=["context", "query"]
#     )
    
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": custom_prompt},
#         memory=memory,
#         verbose=True
#     )
    
#     query_reformulation_prompt = PromptTemplate(
#         input_variables=["question"],
#         template="""Given the following question, please reformulate it to make it more specific and searchable:
        
# Question: {question}
        
# Reformulated question:"""
#     )
    
#     query_reformulation_chain = query_reformulation_prompt | llm
    
#     return {
#         'llm': llm,
#         'retriever': retriever,
#         'reranker': reranker,
#         'qa_chain': qa_chain,
#         'query_reformulation_chain': query_reformulation_chain
#     }

# # Initialize components
# components = initialize_components()

# def extract_key_concepts(docs, top_n=5):
#     all_text = " ".join([doc.page_content for doc in docs])
    
#     stop_words = set(stopwords.words('english'))
#     words = re.findall(r'\b\w+\b', all_text.lower())
#     filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    
#     word_counts = Counter(filtered_words)
#     return word_counts.most_common(top_n)

# def rerank_documents(query, docs, top_k=3):
#     if not docs:
#         return []
    
#     pairs = [[query, doc.page_content] for doc in docs]
#     scores = components['reranker'].predict(pairs)
    
#     scored_docs = list(zip(docs, scores))
#     scored_docs.sort(key=lambda x: x[1], reverse=True)
    
#     return [doc for doc, score in scored_docs[:top_k]]

# @app.post("/query", response_model=dict)
# async def query_rag_system(request: QueryRequest):
#     try:
#         # First, get the reformulated query
#         reformulated_response = components['query_reformulation_chain'].invoke({
#             "question": request.question
#         })
#         reformulated_query = reformulated_response.content
        
#         # Get relevant documents
#         raw_docs = components['retriever'].get_relevant_documents(reformulated_query)
#         reranked_docs = rerank_documents(request.question, raw_docs)
        
#         # Extract and format key concepts
#         key_concepts = extract_key_concepts(reranked_docs)
#         concept_text = "\n".join([f"• {concept}: appears {count} times in the documents" 
#                                 for concept, count in key_concepts])
        
#         # Prepare context
#         retrieved_context = "\n\n".join([doc.page_content for doc in reranked_docs])
#         augmented_context = f"""
# Key Concepts in Retrieved Documents:
# {concept_text}

# Retrieved Information:
# {retrieved_context}
# """
        
#         # Get the answer using the QA chain
#         qa_response = components['qa_chain']({
#             "input_documents": reranked_docs,
#             "query": request.question
#         })
        
#         return {
#             "answer": qa_response["result"],
#             "sources": [doc.metadata.get('source', 'unknown') for doc in reranked_docs]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_core.messages import trim_messages  # Import trim_messages
# from langchain_openai import ChatOpenAI
# from sentence_transformers import CrossEncoder
# from collections import Counter
# import re, os, nltk
# from nltk.corpus import stopwords
# from dotenv import load_dotenv
# from contextlib import asynccontextmanager

# # Download required NLTK data
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)

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
#     print(f"- Vector Store: FAISS")
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
#         raise FileNotFoundError(f"Vector store not found at: {vector_store_path}")
    
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
#     llm = ChatOpenAI(
#         api_key=OPENAI_API_KEY,
#         model=OPENAI_MODEL,
#         temperature=0.2,
#         max_tokens=2000
#     )
    
#     retriever = vector_store.as_retriever(
#         search_type="mmr", 
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 10,
#             "lambda_mult": 0.7
#         }
#     )
    
#     reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
#     custom_prompt = PromptTemplate(
#         template="""
#     Answer the question based on the following context:

#     Context: {context}

#     Question: {question}

#     Additional Instructions:
#     1. If the context doesn't provide enough information, say so instead of making up answers.
#     2. Use specific details from the context to support your answer.
#     3. Keep your answer concise but comprehensive.

#     Answer:
#     """,
#         input_variables=["context", "question"]  # Changed "query" to "question"
#     )
    
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": custom_prompt},
#         verbose=True
#     )
    
#     query_reformulation_prompt = PromptTemplate(
#         input_variables=["question"],
#         template="""Given the following question, please reformulate it to make it more specific and searchable:
        
# Question: {question}
        
# Reformulated question:"""
#     )
    
#     query_reformulation_chain = query_reformulation_prompt | llm
    
#     return {
#         'llm': llm,
#         'retriever': retriever,
#         'reranker': reranker,
#         'qa_chain': qa_chain,
#         'query_reformulation_chain': query_reformulation_chain
#     }

# # Initialize components
# components = initialize_components()

# def extract_key_concepts(docs, top_n=5):
#     all_text = " ".join([doc.page_content for doc in docs])
    
#     stop_words = set(stopwords.words('english'))
#     words = re.findall(r'\b\w+\b', all_text.lower())
#     filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    
#     word_counts = Counter(filtered_words)
#     return word_counts.most_common(top_n)

# def rerank_documents(query, docs, top_k=3):
#     if not docs:
#         return []
    
#     pairs = [[query, doc.page_content] for doc in docs]
#     scores = components['reranker'].predict(pairs)
    
#     scored_docs = list(zip(docs, scores))
#     scored_docs.sort(key=lambda x: x[1], reverse=True)
    
#     return [doc for doc, score in scored_docs[:top_k]]

# @app.post("/query", response_model=dict)
# async def query_rag_system(request: QueryRequest):
#     try:
#         # First, get the reformulated query
#         reformulated_response = components['query_reformulation_chain'].invoke({
#             "question": request.question
#         })
#         reformulated_query = reformulated_response.content
        
#         # Get relevant documents - using invoke() instead of get_relevant_documents()
#         raw_docs = components['retriever'].invoke(reformulated_query)
#         reranked_docs = rerank_documents(request.question, raw_docs)
        
#         # Extract and format key concepts
#         key_concepts = extract_key_concepts(reranked_docs)
#         concept_text = "\n".join([f"• {concept}: appears {count} times in the documents" 
#                                 for concept, count in key_concepts])
        
#         # Prepare context
#         retrieved_context = "\n\n".join([doc.page_content for doc in reranked_docs])
#         augmented_context = f"""
# Key Concepts in Retrieved Documents:
# {concept_text}

# Retrieved Information:
# {retrieved_context}
# """
        
#         # Get the answer using the QA chain with invoke() and correct parameter name
#         qa_response = components['qa_chain'].invoke({
#             "input_documents": reranked_docs,
#             "question": request.question  # Changed from "query" to "question"
#         })
        
#         return {
#             "answer": qa_response["result"],
#             "sources": [doc.metadata.get('source', 'unknown') for doc in reranked_docs]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_core.messages import trim_messages
# from langchain_openai import ChatOpenAI
# from sentence_transformers import CrossEncoder
# from collections import Counter
# import re, os, nltk
# from nltk.corpus import stopwords
# from dotenv import load_dotenv
# from contextlib import asynccontextmanager

# # Download required NLTK data
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)

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
#     print(f"- Vector Store: FAISS")
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
#         raise FileNotFoundError(f"Vector store not found at: {vector_store_path}")
    
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
#     llm = ChatOpenAI(
#         api_key=OPENAI_API_KEY,
#         model=OPENAI_MODEL,
#         temperature=0.2,
#         max_tokens=2000
#     )
    
#     retriever = vector_store.as_retriever(
#         search_type="mmr", 
#         search_kwargs={
#             "k": 5,
#             "fetch_k": 10,
#             "lambda_mult": 0.7
#         }
#     )
    
#     reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
#     # IMPORTANT: Make sure the input variables match what RetrievalQA expects
#     custom_prompt = PromptTemplate(
#         template="""
#     Answer the question based on the following context:

#     Context: {context}

#     Question: {query}

#     Additional Instructions:
#     1. If the context doesn't provide enough information, say so instead of making up answers.
#     2. Use specific details from the context to support your answer.
#     3. Keep your answer concise but comprehensive.

#     Answer:
#     """,
#         input_variables=["context", "query"]  # Changed to use "query" to match what the chain expects
#     )
    
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={
#             "prompt": custom_prompt,
#             # This is key - ensure the input variable name is consistent
#             "input_key": "question"
#         },
#         verbose=True
#     )
    
#     query_reformulation_prompt = PromptTemplate(
#         input_variables=["question"],
#         template="""Given the following question, please reformulate it to make it more specific and searchable:
        
# Question: {question}
        
# Reformulated question:"""
#     )
    
#     query_reformulation_chain = query_reformulation_prompt | llm
    
#     return {
#         'llm': llm,
#         'retriever': retriever,
#         'reranker': reranker,
#         'qa_chain': qa_chain,
#         'query_reformulation_chain': query_reformulation_chain
#     }

# # Initialize components
# components = initialize_components()

# def extract_key_concepts(docs, top_n=5):
#     all_text = " ".join([doc.page_content for doc in docs])
    
#     stop_words = set(stopwords.words('english'))
#     words = re.findall(r'\b\w+\b', all_text.lower())
#     filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    
#     word_counts = Counter(filtered_words)
#     return word_counts.most_common(top_n)

# def rerank_documents(question, docs, top_k=3):
#     if not docs:
#         return []
    
#     pairs = [[question, doc.page_content] for doc in docs]
#     scores = components['reranker'].predict(pairs)
    
#     scored_docs = list(zip(docs, scores))
#     scored_docs.sort(key=lambda x: x[1], reverse=True)
    
#     return [doc for doc, score in scored_docs[:top_k]]

# def invoke_qa_chain(question, context_docs):
#     # Handle parameter name differences between different LangChain versions
#     try:
#         # Since we now know the chain expects 'query', let's try that first
#         return components['qa_chain'].invoke({
#             "query": question,
#             "input_documents": context_docs
#         })
#     except Exception as e:
#         print(f"First attempt failed: {str(e)}")
#         try:
#             # Try alternative parameter names if the first attempt fails
#             context_text = "\n\n".join([doc.page_content for doc in context_docs])
#             return components['qa_chain'].invoke({
#                 "question": question,
#                 "context": context_text
#             })
#         except Exception as e:
#             print(f"Second attempt failed: {str(e)}")
#             raise e

# def create_simple_qa_chain():
#     """Create a minimal QA chain with very explicit parameter naming"""
#     from langchain.chains import StuffDocumentsChain, LLMChain
    
#     # Get components
#     llm = components['llm']
    
#     # Create a prompt that uses 'query' as the parameter name
#     prompt = PromptTemplate(
#         template="""Answer the question based only on the following context:
# Context: {context}
# Question: {query}
# Answer:""",
#         input_variables=["context", "query"]
#     )
    
#     # Create LLM chain that will connect to QA chain
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
    
#     # Create document chain with explicit document variable name
#     qa_chain = StuffDocumentsChain(
#         llm_chain=llm_chain,
#         document_variable_name="context"
#     )
    
#     return qa_chain

# @app.post("/query", response_model=dict)
# async def query_rag_system(request: QueryRequest):
#     try:
#         # Get reformulated query
#         reformulated_response = components['query_reformulation_chain'].invoke({
#             "question": request.question
#         })
#         reformulated_query = reformulated_response.content
        
#         # Get documents using the retriever
#         raw_docs = components['retriever'].invoke(reformulated_query)
#         reranked_docs = rerank_documents(request.question, raw_docs)
        
#         # Create a simple QA chain
#         qa_chain = create_simple_qa_chain()
        
#         # Run the chain with explicit parameters
#         qa_response = qa_chain.invoke({
#             "input_documents": reranked_docs,
#             "query": request.question
#         })
        
#         return {
#             "answer": qa_response["output_text"],
#             "sources": [doc.metadata.get('source', 'unknown') for doc in reranked_docs]
#         }
#     except Exception as e:
#         import traceback
#         print(f"Detailed error: {traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from llama_index.core import PromptHelper
# from llama_index.core import VectorStoreIndex
# from llama_index.llms.openai import OpenAI # Updated import
# from llama_index.core import StorageContext, load_index_from_storage
# from sentence_transformers import CrossEncoder
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
#         max_tokens=2000
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
        
#         # Get documents using the index
#         query_engine = components['index'].as_query_engine(
#             similarity_top_k=10,
#             llm=components['llm']
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


# # vectorize_webpages.py
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader, TextLoader
# import datetime
# import glob
# # Replace your text splitter with a more context-aware version

# def process_documents(folder_path):
#     """
#     Process documents with enhanced metadata extraction.
#     """
#     doc_chunks = load_and_split_documents(folder_path)
    
#     # Add enhanced metadata to each chunk
#     for i, chunk in enumerate(doc_chunks):
#         # Extract filename without extension
#         source_file = os.path.basename(chunk.metadata.get('source', 'unknown'))
#         file_name, _ = os.path.splitext(source_file)
        
#         # Add enhanced metadata
#         chunk.metadata.update({
#             'chunk_id': i,
#             'source_file': source_file,
#             'file_name': file_name,
#             'chunk_size': len(chunk.page_content),
#             'processed_date': datetime.now().isoformat()
#         })
    
#     return doc_chunks



# def get_content_specific_splitter(file_path):
#     """Return a text splitter optimized for specific content types."""
    
#     file_extension = os.path.splitext(file_path)[1].lower()
    
#     if file_extension == '.html':
#         # For HTML, use a splitter that respects HTML structure
#         return RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             separators=["</div>", "</p>", "</li>", "<br>", "\n\n", "\n", ". ", " "],
#             length_function=len
#         )
#     elif file_extension == '.txt':
#         # For plain text, use paragraph-aware splitting
#         return RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=150,
#             separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
#             length_function=len
#         )
#     else:
#         # Default splitter
#         return RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )

# def load_and_split_documents(folder_path):
#     """
#     Load documents and split them using content-specific strategies.
#     """
#     all_docs = []
    
#     # Process all supported file types
#     for file_type, loader_cls in [
#         ("*.html", UnstructuredHTMLLoader),
#         ("*.txt", TextLoader),
#         # Add more file types as needed
#     ]:
#         # Get all files of this type
#         file_paths = glob.glob(os.path.join(folder_path, file_type))
        
#         for file_path in file_paths:
#             # Load the document
#             loader = loader_cls(file_path)
#             doc = loader.load()[0]
            
#             # Get the appropriate splitter for this file type
#             splitter = get_content_specific_splitter(file_path)
            
#             # Split the document
#             chunks = splitter.split_documents([doc])
            
#             # Add to our collection
#             all_docs.extend(chunks)
    
#     print(f"Loaded and split {len(all_docs)} chunk(s) from folder: {folder_path}")
#     return all_docs

# def update_vector_store(new_docs_folder: str, vector_store_path: str):
#     """
#     Update the existing FAISS vector store with new documents.
#     """
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Load the existing vector store
#     if os.path.exists(vector_store_path):
#         vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
#         print("Existing vector store loaded.")
#     else:
#         vector_store = None
#         print("No existing vector store found. Creating a new one.")

#     # Process new documents
#     new_doc_chunks = load_and_split_documents(new_docs_folder)

#     # Add new chunks to the vector store
#     if vector_store:
#         vector_store.add_documents(new_doc_chunks)
#         print(f"Added {len(new_doc_chunks)} new document chunks to the vector store.")
#     else:
#         vector_store = FAISS.from_documents(new_doc_chunks, embeddings)
#         print(f"Created a new vector store with {len(new_doc_chunks)} document chunks.")

#     # Save the updated vector store
#     vector_store.save_local(vector_store_path)
#     print(f"Updated vector store saved to: {vector_store_path}")

# if __name__ == "__main__":
#     # Specify the folder containing your new HTML or TXT files
#     new_docs_folder = "webpages"  # Folder with new webpages or text files

#     # Specify the path to the existing or new vector store
#     vector_store_path = "vector_store"  # Folder where the vector store is saved

#     # Update the vector store with new documents
#     update_vector_store(new_docs_folder, vector_store_path)






# # vectorize_webpages.py
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader, TextLoader
# import datetime
# import glob
# # Replace your text splitter with a more context-aware version

# def process_documents(folder_path):
#     """
#     Process documents with enhanced metadata extraction.
#     """
#     doc_chunks = load_and_split_documents(folder_path)
    
#     # Add enhanced metadata to each chunk
#     for i, chunk in enumerate(doc_chunks):
#         # Extract filename without extension
#         source_file = os.path.basename(chunk.metadata.get('source', 'unknown'))
#         file_name, _ = os.path.splitext(source_file)
        
#         # Add enhanced metadata
#         chunk.metadata.update({
#             'chunk_id': i,
#             'source_file': source_file,
#             'file_name': file_name,
#             'chunk_size': len(chunk.page_content),
#             'processed_date': datetime.datetime.now().isoformat(),
#             'language': 'de' if 'german' in file_name.lower() else 'en',  # Example: detect language
#             'type': 'faq' if '#Frage:' in chunk.page_content else 'general'
#         })
    
#     return doc_chunks


# def get_content_specific_splitter(file_path):
#     """Return a text splitter optimized for specific content types."""
    
#     file_extension = os.path.splitext(file_path)[1].lower()
    
#     if file_extension == '.html':
#         # For HTML, use a splitter that respects HTML structure
#         return RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             separators=["</div>", "</p>", "</li>", "<br>", "\n\n", "\n", ". ", " "],
#             length_function=len
#         )
#     elif file_extension == '.txt':
#         # For plain text FAQs, use question-answer aware splitting
#         return RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=150,
#             separators=["#Frage:", "#Antwort:", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
#             length_function=len
#         )
#     else:
#         # Default splitter
#         return RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )

# def load_and_split_documents(folder_path):
#     """
#     Load documents and split them using content-specific strategies.
#     """
#     all_docs = []
    
#     # Process all supported file types
#     for file_type, loader_cls in [
#         ("*.html", UnstructuredHTMLLoader),
#         ("*.txt", TextLoader),
#         # Add more file types as needed
#     ]:
#         # Get all files of this type
#         file_paths = glob.glob(os.path.join(folder_path, file_type))
        
#         for file_path in file_paths:
#             # Load the document
#             loader = loader_cls(file_path)
#             doc = loader.load()[0]
            
#             # Get the appropriate splitter for this file type
#             splitter = get_content_specific_splitter(file_path)
            
#             # Split the document
#             chunks = splitter.split_documents([doc])
            
#             # Add to our collection
#             all_docs.extend(chunks)
    
#     print(f"Loaded and split {len(all_docs)} chunk(s) from folder: {folder_path}")
#     return all_docs

# def update_vector_store(new_docs_folder: str, vector_store_path: str):
#     """
#     Update the existing FAISS vector store with new documents.
#     """
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Load the existing vector store
#     if os.path.exists(vector_store_path):
#         vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
#         print("Existing vector store loaded.")
#     else:
#         vector_store = None
#         print("No existing vector store found. Creating a new one.")

#     # Process new documents
#     new_doc_chunks = load_and_split_documents(new_docs_folder)

#     # Add new chunks to the vector store
#     if vector_store:
#         vector_store.add_documents(new_doc_chunks)
#         print(f"Added {len(new_doc_chunks)} new document chunks to the vector store.")
#     else:
#         vector_store = FAISS.from_documents(new_doc_chunks, embeddings)
#         print(f"Created a new vector store with {len(new_doc_chunks)} document chunks.")

#     # Save the updated vector store
#     vector_store.save_local(vector_store_path)
#     print(f"Updated vector store saved to: {vector_store_path}")

# if __name__ == "__main__":
#     # Specify the folder containing your new HTML or TXT files
#     new_docs_folder = "webpages"  # Folder with new webpages or text files

#     # Specify the path to the existing or new vector store
#     vector_store_path = "vector_store"  # Folder where the vector store is saved

#     # Update the vector store with new documents
#     update_vector_store(new_docs_folder, vector_store_path)


