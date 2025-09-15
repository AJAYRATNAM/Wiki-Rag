# RAG Chatbot Project

## Project Contents

This project demonstrates multiple approaches to building a **Retrieval-Augmented Generation (RAG) chatbot** using Python, LangChain, HuggingFace models, and hybrid search techniques. It contains the following scripts:

- [VANILLA_RAG](https://github.com/AJAYRATNAM/Wiki-Rag/blob/main/notebooks/Vanilla_RAG.ipynb) – A basic RAG pipeline using a vector store for dense retrieval and a local HuggingFace model for generation.  
- [HYBRID_SEARCH](https://github.com/AJAYRATNAM/Wiki-Rag/blob/main/notebooks/Hybrid_search.ipynb) – Combines dense retrieval (vector embeddings) with sparse retrieval (BM25) for improved context coverage.  
- [HYBRID_SEARCH+RERANKER](https://github.com/AJAYRATNAM/Wiki-Rag/blob/main/notebooks/Hybrid_search_reranker2.ipynb) – Extends hybrid retrieval with a Cross-Encoder reranker to prioritize the most relevant documents.  
- [HYBRID_SEARCH+RERANK+MULTI_QUERY_RETRIEVAL](https://github.com/AJAYRATNAM/Wiki-Rag/blob/main/notebooks/Hybrid_search_reranker2_multi_query.ipynb) – Further extends the pipeline with multi-query retrieval for robust handling of diverse questions.  

Each version builds on the previous one, incrementally improving retrieval quality and answer accuracy.

---

## Approaches Tried

### 1. Vanilla RAG (`VANILLA_RAG`)
- **Data Source:** Wikipedia page (e.g., Kaggle).  
- **Text Processing:** Split documents into chunks using `RecursiveCharacterTextSplitter`.  
- **Embedding & Retrieval:** Uses `HuggingFaceEmbeddings` and `Chroma` vectorstore for dense retrieval.  
- **LLM:** Local HuggingFace pipeline with `google/flan-t5-base`.  
- **Prompting:** Template ensures the model only answers using provided context, replying `"I don't know"` if insufficient information exists.

### 2. Hybrid Search (`HYBRID_SEARCH`)
- Adds **BM25 sparse retrieval** to the dense vector retrieval.  
- Implements `hybrid_retriever` to merge dense and sparse results (deduplicated).  
- Uses a weighted approach (`alpha`) to combine dense and sparse scores (optional tuning).  
- Aims to reduce missing answers for fact-based questions.

### 3. Hybrid Search + Reranker (`HYBRID_SEARCH+RERANKER`)
- Integrates **Cross-Encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve top-k document relevance.  
- **Steps:**
  1. Retrieve top documents using hybrid retrieval.  
  2. Rerank them based on semantic similarity with the query.  
- Improves answer precision for complex or ambiguous queries.

### 4. Hybrid Search + Reranker + Multi-Query Retrieval (`HYBRID_SEARCH+RERANK+MULTI_QUERY_RETRIEVAL.py`)
- Extends reranked hybrid retrieval to support **multi-query retrieval**, improving context coverage.  
- Ensures that questions with multiple aspects or numbers are more likely to retrieve all relevant context.  
- Maintains the same LLM pipeline and prompt design for consistency.

---




