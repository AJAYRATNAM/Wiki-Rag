from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import config
from langchain_core.runnables import RunnablePassthrough

# --- Model configs ---
EMBEDDING_MODEL = config.EMBEDDING_MODEL
LLM_MODEL = config.LLM_MODEL
RERANKER_MODEL = config.RERANKER_MODEL
REPHRASER_MODEL = config.REPHRASER_MODEL
TOP_K = config.TOP_K


# --- LLM + prompt setup ---
template = """
You are a QA assistant.
Rephrase the Answer only using the provided context.
if context is irrelevant to question, reply "I don't know".
If multiple numbers or facts are present, prefer the most recent one.



Context:
{context}

Question: {question}

Answer:
"""



prompt = PromptTemplate(template=template, input_variables=["context", "question"])
generator = pipeline("text2text-generation", model=LLM_MODEL, max_length=1024, truncation=True)
llm = HuggingFacePipeline(pipeline=generator)
reranker = CrossEncoder(RERANKER_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# -----------------------------
# Helper functions
# -----------------------------
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

def bm25_retriever(query,text_chunks,k):
    corpus = [doc.page_content for doc in text_chunks]
    vectorizer = CountVectorizer().build_tokenizer()
    tokenized_corpus = [vectorizer(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = vectorizer(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-k:][::-1]
    top_docs = [text_chunks[i] for i in top_indices]
    return top_docs

def rerank(query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]


def generate_query_variations(query, paraphraser, num_variations=3):
    prompt = f"paraphrase: {query}"
    results = paraphraser(prompt, max_new_tokens=64, num_return_sequences=num_variations, do_sample=True)
    variations = [r["generated_text"].strip() for r in results]
    return [query] + variations

def multi_query_retriever(query, retriever,paraphraser,text_chunks,k):
    # Step 1: Expand queries
    variations = generate_query_variations(query, paraphraser)

    all_docs = []
    for q in variations:
        # Run hybrid retriever for each variation
        docs = hybrid_retriever(q, retriever,text_chunks,k)
        all_docs.extend(docs)

    # Step 2: Deduplicate
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()

    # Step 3: Rerank final set
    reranked_docs = rerank(query, list(unique_docs))
    return reranked_docs[:k]

def hybrid_retriever(query,retriever,text_chunks,k,alpha=0.5):
    """
    alpha: weight for dense vs sparse
    """
    # Get dense results
    dense_docs = retriever.get_relevant_documents(query)
    # Get sparse results
    sparse_docs = bm25_retriever(query,text_chunks,k)

    # Merge & deduplicate (you can tune merging strategy)
    combined_docs = {doc.page_content: doc for doc in dense_docs + sparse_docs}
    merged_docs = list(combined_docs.values())[:k]  # keep top k
    reranked_docs = rerank(query, merged_docs)
    return reranked_docs[:k]

# -----------------------------
# Dynamic RAG per URL
# -----------------------------
def query_dynamic_rag(url, query, top_k=TOP_K):
    # 1. Load web page
    loader = WebBaseLoader(web_paths=[url])
    documents = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #Recurssive splitter for dense context splitting
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings)

    # 3. Create vectorstore in-memory
    vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory=None)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    paraphraser = pipeline("text2text-generation",model=REPHRASER_MODEL)

    final_docs = multi_query_retriever(query,retriever,paraphraser,text_chunks,TOP_K)

    # 5. Query LLM
    context = format_docs(final_docs)
    rag_pipeline = (
    {"context": lambda q: format_docs(multi_query_retriever(query, retriever,paraphraser,text_chunks,top_k)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    answer= rag_pipeline.invoke(query)
    return answer