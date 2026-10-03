class RAGPipelineError(Exception):
    """Base exception for all RAG Pipeline errors."""
    pass

class DocumentIngestionError(RAGPipelineError):
    """Raised when PDF loading, parsing, or chunking fails."""
    pass

class VectorStoreError(RAGPipelineError):
    """Raised when FAISS indexing or retrieval operations fail."""
    pass

