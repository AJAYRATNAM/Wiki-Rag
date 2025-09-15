
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer

# 1. Using weburl as input data
loader = WebBaseLoader(web_paths=["https://en.wikipedia.org/wiki/Kaggle"])
documents = loader.load()

# 2. Splitting into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #Recurssive splitter for dense context splitting
text_chunks = text_splitter.split_documents(documents)

# 3. Embedding with HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings)

#4. Using vectorstore as retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 3})

template = """
You are a QA assistant.
Answer only using the provided context.
if context is irrelevant to question, reply "I don't know".
If multiple numbers or facts are present, prefer the most recent one.



Context:
{context}

Question: {question}

Answer:
"""

#5. Prompt template created for model to not utitlize its pretrained data
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# 5. LLM (local HuggingFace model)
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, truncation=True)
llm = HuggingFacePipeline(pipeline=generator)


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# Convert your text chunks into plain text for BM25
corpus = [doc.page_content for doc in text_chunks]

# Tokenize
vectorizer = CountVectorizer().build_tokenizer()
tokenized_corpus = [vectorizer(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

# Function to get top k sparse results
def bm25_retriever(query, k=3):
    tokenized_query = vectorizer(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-k:][::-1]
    top_docs = [text_chunks[i] for i in top_indices]
    return top_docs


def hybrid_retriever(query,retriever,k=3, alpha=0.5):
    """
    alpha: weight for dense vs sparse
    """
    # Get dense results
    dense_docs = retriever.get_relevant_documents(query)
    # Get sparse results
    sparse_docs = bm25_retriever(query, k=k)

    # Merge & deduplicate (you can tune merging strategy)
    combined_docs = {doc.page_content: doc for doc in dense_docs + sparse_docs}
    merged_docs = list(combined_docs.values())[:k]  # keep top k
    return merged_docs

rag_pipeline = (
    {"context": lambda q: format_docs(hybrid_retriever(q,retriever)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_pipeline.invoke("What is the use of Kaggle"))

print(rag_pipeline.invoke("what is python"))

print(rag_pipeline.invoke("how to transfer money"))

print(rag_pipeline.invoke("When was Kaggle launched?"))

print(rag_pipeline.invoke("When was Java launched?"))

print(rag_pipeline.invoke("What is oops concept?"))

print(rag_pipeline.invoke("why Kaggle was so famous?"))

print(rag_pipeline.invoke("What is a Kaggle Grandmaster?"))

print(rag_pipeline.invoke("How many Kaggle users are there?"))

print(rag_pipeline.invoke("how many people using kaggle"))