import os

# -----------------------------
# RAG Model Configs
# -----------------------------
# Dense embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM model for answer generation
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

# Reranker model
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

REPHRASER_MODEL = os.getenv("REPHRASER_MODEL","humarin/chatgpt_paraphraser_on_T5_base")  # Model for generating paraphrased queries



# -----------------------------
# Retriever / Hybrid Configs
# -----------------------------
# Top K documents to retrieve from dense/sparse retrievers
TOP_K = int(os.getenv("TOP_K", 3))

# BM25 vectorizer settings (if needed)
BM25_TOKENIZER = None  # Can set a custom tokenizer here

# -----------------------------
# Text Splitting / Chunking
# -----------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))         # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))    # overlap between chunks



# -----------------------------
# Misc
# -----------------------------
# Temporary directory for in-memory vectorstore (optional caching)
TEMP_VECTORSTORE_DIR = os.getenv("TEMP_VECTORSTORE_DIR", None)

# Debug flag
DEBUG = os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"]
