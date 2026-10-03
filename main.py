import logging
import argparse
from src.config import settings
from src.ingestion import load_and_chunk_documents
from src.vector_store import VectorStoreManager
from src.rag_chain import build_rag_chain

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)

def ingest():
    """Execute the data ingestion phase."""
    chunks = load_and_chunk_documents(
        data_dir=settings.data_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    if chunks:
        vsm = VectorStoreManager(settings.embedding_model_id)
        vsm.build_and_save_index(chunks, settings.db_path)
    else:
        logger.error("No chunks generated. Ensure valid PDFs exist in data/docs/")

def chat():
    """Initialize the RAG backend and start an interactive prompt."""
    if not settings.db_path.exists():
        logger.error("FAISS index not found. Please run '--ingest' first.")
        return
        
    vsm = VectorStoreManager(settings.embedding_model_id)
    vector_store = vsm.load_index(settings.db_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})
    
    rag_chain = build_rag_chain(retriever, settings.llm_model_id)
    
    print("\n" + "="*60)
    print("🧠 Local RAG System Online | Type 'exit' or 'quit' to terminate.")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("\n👤 User: ")
            if query.lower() in ['quit', 'exit']:
                break
                
            response = rag_chain.invoke({"input": query})
            print(f"\n🤖 Assistant: {response['answer']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Generation error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Pipeline Backend")
    parser.add_argument("--ingest", action="store_true", help="Parse PDFs and build the FAISS index")
    parser.add_argument("--chat", action="store_true", help="Launch the local terminal chat interface")
    args = parser.parse_args()
    
    if args.ingest:
        ingest()
    elif args.chat:
        chat()
    else:
        parser.print_help()
