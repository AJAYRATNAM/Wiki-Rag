#!pip install langchain_community langchainhub chromadb langchain langchain-openai

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
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
import os
from google.colab import userdata
from datasets import Dataset

os.environ['OPENAI_API_KEY']= userdata.get('chatbot')
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
You are a question answering assistant.

ONLY use the information provided in the context to answer the question.
If the context does not contain the answer, reply exactly: "I don't know".


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


# 6. RAG pipeline
rag_pipeline = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)




