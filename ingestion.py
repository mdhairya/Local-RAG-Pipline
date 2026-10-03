import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.exceptions import DocumentIngestionError

logger = logging.getLogger(__name__)

def load_and_chunk_documents(data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Loads PDFs from a directory and splits them into logical textual chunks."""
    try:
        logger.info(f"Loading PDFs from {data_dir}")
        loader = PyPDFDirectoryLoader(str(data_dir))
        docs = loader.load()
        
        if not docs:
            logger.warning("No documents found in the specified directory.")
            return []
            
        logger.info(f"Loaded {len(docs)} pages. Initializing chunking...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} document chunks.")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise DocumentIngestionError(f"Failed to ingest documents: {e}") from e
