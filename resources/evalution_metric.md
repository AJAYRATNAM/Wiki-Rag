Retrieval Quality Metrics

How well the retriever finds relevant context.

Precision@K / Recall@K → Fraction of retrieved documents that are relevant (precision) vs fraction of relevant documents retrieved (recall).
Example: If only 2 of top-5 chunks are relevant, Precision@5 = 0.4.

Mean Reciprocal Rank (MRR) → Measures how high the first relevant document appears in the ranked list.

Normalized Discounted Cumulative Gain (nDCG) → Accounts for ranking quality by giving higher weight to relevant docs appearing earlier.

🔹 2. Answer Quality Metrics

How good the generated response is.

Faithfulness (Groundedness) → Is the answer supported by retrieved evidence (avoids hallucinations)?

Answer Relevancy (Correctness) → Does the answer address the user’s query meaningfully?

Factual Consistency → Matches ground-truth facts (can be evaluated via NLI models).

Completeness → Does the answer cover all necessary parts of the query?

🔹 3. Context Utilization Metrics

How well the LLM used the retrieved documents.

Context Recall → % of relevant ground-truth facts covered in the answer.

Context Precision → % of answer content actually supported by retrieved context.

Context Utilization Rate → How often retrieved context actually influences the final answer.

🔹 4. User-Centric / Utility Metrics

How the system feels in practice.

Helpfulness / Usefulness (LLM-as-judge or human eval) → Would the answer help a real user?

Readability / Fluency → Grammar, style, coherence.

Toxicity / Bias → Is the response safe and unbiased?

Latency / Efficiency → Time taken to generate an answer.

🔹 5. System-Level Metrics

For production RAG evaluation.

Coverage → % of queries where at least one relevant doc is retrieved.

Answerability → % of queries where the system should return “I don’t know” but doesn’t hallucinate.

Calibration → Does model express uncertainty correctly (confidence vs correctness)?

Cost Efficiency → Tokens used per query (retrieval + generation).

✅ Interview Tip:
If asked “How would you evaluate a RAG system?”, structure your answer in layers:

Retrieval quality → Precision, Recall, MRR, nDCG.

Generation quality → Faithfulness, relevancy, factual consistency, completeness.

Context usage → Context recall/precision, utilization.

User/system metrics → Helpfulness, safety, latency, cost.

This shows you think end-to-end instead of just “answer looks good”.