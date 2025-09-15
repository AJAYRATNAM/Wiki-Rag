reate dataset in the correct format
data = {
    "question": [
        "What is Kaggle?",
        "When was Kaggle founded?",
        "Who founded Kaggle?",
        "Who owns Kaggle now?",
        "What type of competitions are hosted on Kaggle?",
        "What is Kaggle Kernels?",
        "When did Google acquire Kaggle?",
        "What programming languages are mainly used on Kaggle?",
        "What can Kaggle users do besides competitions?",
        "Where is Kaggle headquartered?"
    ],
    "answer": [
        rag_pipeline.invoke("What is Kaggle?"),
        rag_pipeline.invoke("When was Kaggle founded?"),
        rag_pipeline.invoke("Who founded Kaggle?"),
        rag_pipeline.invoke("Who owns Kaggle now?"),
        rag_pipeline.invoke("What type of competitions are hosted on Kaggle?"),
        rag_pipeline.invoke("What is Kaggle Kernels?"),
        rag_pipeline.invoke("When did Google acquire Kaggle?"),
        rag_pipeline.invoke("What programming languages are mainly used on Kaggle?"),
        rag_pipeline.invoke("What can Kaggle users do besides competitions?"),
        rag_pipeline.invoke("Where is Kaggle headquartered?")
    ],
    "contexts": [
        [c.page_content for c in retriever.get_relevant_documents("What is Kaggle?")],
        [c.page_content for c in retriever.get_relevant_documents("When was Kaggle founded?")],
        [c.page_content for c in retriever.get_relevant_documents("Who founded Kaggle?")],
        [c.page_content for c in retriever.get_relevant_documents("Who owns Kaggle now?")],
        [c.page_content for c in retriever.get_relevant_documents("What type of competitions are hosted on Kaggle?")],
        [c.page_content for c in retriever.get_relevant_documents("What is Kaggle Kernels?")],
        [c.page_content for c in retriever.get_relevant_documents("When did Google acquire Kaggle?")],
        [c.page_content for c in retriever.get_relevant_documents("What programming languages are mainly used on Kaggle?")],
        [c.page_content for c in retriever.get_relevant_documents("What can Kaggle users do besides competitions?")],
        [c.page_content for c in retriever.get_relevant_documents("Where is Kaggle headquartered?")]
    ],
    "reference": [
        "Kaggle is an online community and platform for data scientists and machine learning practitioners.",
        "Kaggle was founded in 2010.",
        "Kaggle was founded by Anthony Goldbloom and Ben Hamner.",
        "Kaggle is owned by Google LLC.",
        "Kaggle hosts data science and machine learning competitions.",
        "Kaggle Kernels is a cloud-based code execution environment that allows users to run Jupyter notebooks on Kaggle.",
        "Google acquired Kaggle in March 2017.",
        "Python and R are the main programming languages used on Kaggle.",
        "Besides competitions, Kaggle users can share datasets, publish notebooks, and collaborate with others.",
        "Kaggle is headquartered in San Francisco, California."
    ]
}

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
print(result)