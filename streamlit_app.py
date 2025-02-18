import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to load the pre-built vector store
def load_vector_store(vector_store_path):
    """
    Load the FAISS vector store from disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Streamlit App
def main():
    st.title("RAG System for Waldorf 365 📚")
    st.write("Ask questions about the pre-processed documents!")

    # Path to the pre-built vector store
    vector_store_path = "vector_store"  # Update to your vector store path

    # Load the vector store
    if os.path.exists(vector_store_path):
        vector_store = load_vector_store(vector_store_path)
        st.success("Vector store loaded successfully!")
    else:
        st.error("Vector store not found. Please run the 'vectorize_webpages.py' script first.")
        return

    # Initialize Groq LLM and RetrievalQA chain
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Question input
    query = st.text_input("Ask a question about the uploaded documents:")

    if query:
        # Get answer
        with st.spinner("Generating answer..."):
            answer = qa_chain.invoke(query)
        st.subheader("Answer:")
        st.write(answer)

# Run the app
if __name__ == "__main__":
    main()