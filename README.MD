# 🧠 Local RAG Pipeline

A completely localized, production-ready Retrieval-Augmented Generation (RAG) backend. This project utilizes PyTorch-based Hugging Face embeddings, FAISS for high-performance vector search, and modern LangChain LCEL architectures to chat directly with local PDF documents. No data leaves your machine.

## 🚀 Features
- **100% Offline/Local**: Total data privacy.
- **Modern LCEL Design**: Built with LangChain's latest expression language.
- **Robust Configuration**: Environment management via Pydantic.
- **Custom Error Handling**: Clean decoupling of IO and logic for production.

## 📦 Setup & Installation

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
