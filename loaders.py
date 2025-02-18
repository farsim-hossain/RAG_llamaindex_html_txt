
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader


def load_documents_from_folder(folder_path: str):
    """
    Load all HTML files from the specified folder into a list of Documents.
    """
    # The glob pattern "*.html" ensures only HTML files are processed.
    loader = DirectoryLoader(folder_path, glob="*.html", loader_cls=UnstructuredHTMLLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from folder: {folder_path}")
    return docs
