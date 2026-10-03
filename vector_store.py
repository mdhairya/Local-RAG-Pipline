import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the FAISS lifecycle and Hugging Face Embeddings."""
    
    def __init__(self, embedding_model_id: str):
        logger.info(f"Initializing embedding model: {embedding_model_id}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_id,
            model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU is available
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def build_and_save_index(self, chunks: list[Document], save_path: Path) -> FAISS:
        """Embeds document chunks into a FAISS index and persists it."""
        try:
            logger.info("Embedding chunks and building FAISS index...")
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(save_path))
            logger.info(f"FAISS index successfully saved to {save_path}")
            return vector_store
        except Exception as e:
            raise VectorStoreError(f"Failed to build vector store: {e}") from e
            
    def load_index(self, load_path: Path) -> FAISS:
        """Loads an existing FAISS index from disk securely."""
        try:
            logger.info(f"Loading FAISS index from {load_path}")
            return FAISS.load_local(
                str(load_path), 
                self.embeddings,
                allow_dangerous_deserialization=True # Required for local, trusted pickle loading
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {e}") from e
