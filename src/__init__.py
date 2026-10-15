"""Local PDF QA core package.

Modules:
- ollama_client: minimal HTTP client for the local Ollama server
- database:       SQLite storage (documents, chunks, FTS5, embeddings)
- pdf_processor:  PyMuPDF text extraction with robust error handling
- chunker:        paragraph/sentence-aware text chunking
- embeddings:     local batch embedding generation via Ollama
- retriever:      hybrid semantic (NumPy) + keyword (FTS5) retrieval with RRF
- answer_generator: strictly grounded answer generation with citations
- ingestion:      end-to-end incremental ingestion pipeline
"""

__all__ = [
    "ollama_client",
    "database",
    "pdf_processor",
    "chunker",
    "embeddings",
    "retriever",
    "answer_generator",
    "ingestion",
]
