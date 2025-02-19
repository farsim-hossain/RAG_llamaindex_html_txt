
import os
import glob
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    Document
)

from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, load_index_from_storage
import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(api_key=OPENAI_API_KEY)

def process_documents(folder_path):
    """
    Process documents with enhanced metadata extraction.
    """
    documents = SimpleDirectoryReader(input_dir=folder_path).load_data()
    all_docs = []

    # Initialize the SentenceSplitter with desired parameters
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    for doc in documents:
        file_path = doc.metadata['file_path']  # Changed from extra_info to metadata
        file_extension = os.path.splitext(file_path)[1].lower()
        nodes = splitter.get_nodes_from_documents([doc])

        for i, node in enumerate(nodes):
            source_file = os.path.basename(file_path)
            file_name, _ = os.path.splitext(source_file)

            # Convert node to Document
            doc_text = node.get_text()
            doc_metadata = {
                'chunk_id': i,
                'source_file': source_file,
                'file_name': file_name,
                'chunk_size': len(doc_text),
                'processed_date': datetime.datetime.now().isoformat(),
                'language': 'de' if 'german' in file_name.lower() else 'en',
                'type': 'faq' if '#Frage:' in doc_text else 'general'
            }
            document = Document(text=doc_text, metadata=doc_metadata)
            all_docs.append(document)

    print(f"Loaded and split {len(all_docs)} chunk(s) from folder: {folder_path}")
    return all_docs

def update_vector_store(new_docs_folder: str, vector_store_path: str):
    """
    Update the existing LlamaIndex vector store with new documents.
    """
    Settings.llm = OpenAI(api_key=OPENAI_API_KEY)
    Settings.context_window = 4096
    Settings.num_output = 256

    # Load existing index
    if os.path.exists(vector_store_path):
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=vector_store_path
            )
            index = load_index_from_storage(storage_context)
            print("Existing index loaded.")
        except Exception as e:
            print(f"Error loading index: {e}")
            index = None
    else:
        index = None
        print("No existing index found. Creating a new one.")

    # Process new documents
    new_docs = process_documents(new_docs_folder)

    # Create or update index
    if index:
        for doc in new_docs:
            index.insert(doc)
        print(f"Added {len(new_docs)} new documents to the index.")
    else:
        index = VectorStoreIndex.from_documents(new_docs)
        print(f"Created a new index with {len(new_docs)} documents.")

    # Save index
    if not os.path.exists(vector_store_path):
        os.makedirs(vector_store_path)
    
    index.storage_context.persist(persist_dir=vector_store_path)
    print(f"Index persisted to: {vector_store_path}")

if __name__ == "__main__":
    new_docs_folder = "webpages"
    vector_store_path = "vector_store"
    update_vector_store(new_docs_folder, vector_store_path)
