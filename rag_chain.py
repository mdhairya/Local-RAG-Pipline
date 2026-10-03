import logging
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

def build_rag_chain(retriever: VectorStoreRetriever, model_id: str):
    """Constructs the Retrieval-Augmented Generation chain using LCEL."""
    logger.info(f"Booting local LLM pipeline with: {model_id}")
    
    # Configure the PyTorch backend pipeline
    hf_pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=256,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    system_prompt = (
        "You are an expert internal assistant. Use the provided context to answer the user's question.\n"
        "If the answer is not contained within the context, explicitly state that you do not know.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    logger.info("Composing the LCEL retrieval chain...")
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    return rag_chain
