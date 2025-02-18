
from loaders import load_documents_from_folder
from rag_chain import build_rag_chain

def main():
    # Specify the folder containing your downloaded webpages (HTML files)
    folder_path = "webpages"  # Update to your folder path
    
    # Load all documents from the folder
    docs = load_documents_from_folder(folder_path)
    
    # Build the RAG system from the loaded documents
    qa_chain = build_rag_chain(docs)
    
    # Run a sample query against the RAG system
    query = "What is the main topic of these webpages?"
    answer = qa_chain.invoke(query)  
    print("Answer:", answer)

if __name__ == "__main__":
    main()
