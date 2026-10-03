from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Centralized configuration for the RAG Pipeline."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data" / "docs"
    db_path: Path = base_dir / "data" / "faiss_index"
    
    # Model Configurations
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Tip: Use 'mistralai/Mistral-7B-Instruct-v0.2' if you have GPU, 'gpt2' for basic CPU testing
    llm_model_id: str = "gpt2" 
    
    # Ingestion Configurations
    chunk_size: int = 1000
    chunk_overlap: int = 150
    
    # Retrieval Configurations
    retriever_k: int = 3
    
    class Config:
        env_file = ".env"

settings = Settings()
