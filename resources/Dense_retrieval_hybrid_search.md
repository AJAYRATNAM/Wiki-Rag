# Retrieval Methods in RAG and Search Systems

---

## 1️⃣ Sparse Retrieval

**Definition:**  
Relies on term-level matching between the query and documents (like traditional search engines).  
Uses inverted indices (like Elasticsearch, BM25, TF-IDF).

**How it works:**  
- Every document is represented as a bag of words (sparse vector).  
- When a query comes, you look for documents that **share the same words**.

**Example:**  
- Query: `"Kaggle machine learning competitions"`  
- BM25 will rank documents with **exact matches** of `"Kaggle"`, `"machine"`, `"learning"`, `"competitions"` higher.

**Pros:**  
- Fast, interpretable, easy to implement.  
- Works well for queries with exact keywords.

**Cons:**  
- Misses semantic matches (paraphrases).  
- Cannot handle synonyms, e.g., `"ML contests"` vs `"machine learning competitions"`.

**Common libraries:** BM25, TF-IDF, Elasticsearch, Whoosh

---

## 2️⃣ Dense Retrieval

**Definition:**  
Relies on **vector embeddings** to capture semantic meaning of queries and documents.  
Uses neural networks (like BERT, Sentence-BERT) to embed text into dense vectors.

**How it works:**  
1. Encode all documents into vectors (`d1, d2, ...`).  
2. Encode query into vector `q`.  
3. Retrieve documents by **vector similarity** (e.g., cosine similarity, dot product).

**Example:**  
- Query: `"ML contests"`  
- Even if the document says `"machine learning competitions"`, dense retrieval can match because embeddings capture **semantic similarity**.

**Pros:**  
- Captures synonyms, paraphrases, and context.  
- Better recall for open-ended queries.

**Cons:**  
- Slower (especially for huge corpora).  
- Needs GPU for embedding large datasets efficiently.  
- Less interpretable than sparse retrieval.

**Common libraries:** FAISS, Chroma, Milvus, Weaviate, HuggingFace Sentence-Transformers

---

## 3️⃣ Hybrid Approach

Combine sparse + dense retrieval for the **best of both worlds**:  

- Sparse → ensures **keyword match**  
- Dense → captures **semantic similarity**  

Often implemented as **score fusion** or **two-stage retrieval**.

---

## BM25 (Best Matching 25)

**Definition:**  
BM25 is a ranking function used in information retrieval to score documents based on their **relevance to a query**.  
It improves on TF-IDF by considering **term frequency saturation** and **document length normalization**.

### Term Frequency (TF)  
- Measures how often a query term appears in a document.  
- Uses a **saturation function** so repeated words don’t increase score linearly.

### Inverse Document Frequency (IDF)  
- Rare terms are more informative and carry more weight.

### Document Length Normalization (DLN)  
- Avoids bias toward long documents.  

**How BM25 Works:**  
1. Compute **term-level scores** using TF, IDF, and length normalization for each document.  
2. Sum the scores for all query terms.  
3. Rank documents by this score.

---

## Document Length Normalization (DLN)

**Problem it solves:**  
- In TF-IDF or term-frequency–based scoring, **long documents tend to score higher** just because they have more words.

**Example:**  
- Short doc: `"Kaggle hosts ML competitions"` → 3 query terms  
- Long doc: `"Kaggle hosts ML competitions. Data science tutorials. Python coding."` → 3 query terms + extra words  

Without normalization, the long doc might score higher simply because it has **more words**, not because it’s more relevant.
