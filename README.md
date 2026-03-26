# PDF RAG Chatbot

A local RAG system to chat with PDF documents.

## Stack
-**PyMuPDF** - PDF text extraction
-**sentence-trasformers** - text embeddings
-**ChromaDB** - vector storage
-**Ollama** - local LLM 

## Setup
1. Install dependencies:
 pip install pymupdf sentence-transformers chromadb ollama

2. Install and run Ollama from https://ollama.com
 ollama pull llama3

3. Add your PDF to the project folder

4. Run:
   python Ingestion.py