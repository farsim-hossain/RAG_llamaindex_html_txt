
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from config import GROQ_API_KEY

def build_rag_chain(docs):
    """
    Build a RetrievalQA chain from documents using Groq LLM and Hugging Face embeddings.
    """
    # 1. Split documents into chunks (with overlap to preserve context)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)
    
    # 2. Create embeddings using a free Hugging Face model.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Build a FAISS vector store from the document chunks.
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    
    # 4. Initialize the Groq LLM.
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    
    # 5. Create a RetrievalQA chain (using the "stuff" method to concatenate retrieved chunks).
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qa_chain
