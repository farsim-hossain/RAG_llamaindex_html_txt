

import os
from google.colab import files  # For file uploads and downloads in Colab
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader, TextLoader

def load_and_split_documents(folder_path):
    """
    Load documents from the folder (both HTML and TXT) and split them into chunks.
    """
    # Load HTML files
    html_loader = DirectoryLoader(folder_path, glob="*.html", loader_cls=UnstructuredHTMLLoader)
    html_docs = html_loader.load()

    # Load TXT files
    txt_loader = DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()

    # Combine all documents
    all_docs = html_docs + txt_docs
    print(f"Loaded {len(all_docs)} document(s) from folder: {folder_path}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(all_docs)
    return doc_chunks

def create_vector_store(docs_folder: str, output_path: str):
    """
    Create a FAISS vector store from the documents in the specified folder.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Process documents
    doc_chunks = load_and_split_documents(docs_folder)

    # Build FAISS vector store
    vector_store = FAISS.from_documents(doc_chunks, embeddings)

    # Save the vector store locally
    vector_store.save_local(output_path)
    print(f"Vector store saved to: {output_path}")

# Main execution in Colab
if __name__ == "__main__":
    # Step 1: Upload your files to Colab
    print("Upload your HTML and TXT files to Colab...")
    uploaded = files.upload()  # This allows you to upload files via the browser

    # Create a folder for the uploaded files
    docs_folder = "webpages"
    os.makedirs(docs_folder, exist_ok=True)

    # Save uploaded files to the folder
    for filename, content in uploaded.items():
        file_path = os.path.join(docs_folder, filename)
        with open(file_path, "wb") as f:
            f.write(content)
    print(f"Saved {len(uploaded)} file(s) to folder: {docs_folder}")

    # Step 2: Create the vector store
    output_path = "vector_store"
    create_vector_store(docs_folder, output_path)

    # Step 3: Download the vector store
    print("Downloading the vector store...")
    files.download(f"{output_path}/index.faiss")
    files.download(f"{output_path}/index.pkl")