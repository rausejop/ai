# Chapter 5.5: Information Retrieval: Question Answering Systems

## 1. Extractive QA (SQuAD)
**Extractive Question Answering** is a specialized task where the answer is known to be a specific span of text within a provided document. Given a paragraph and a question, the model's objective is to predict the **Start Token Index** and the **End Token Index** of the correct answer. This task was popularized by the **SQuAD** (Stanford Question Answering Dataset) benchmark and is the fundamental technology behind automated FAQ searching and technical manual navigation.

## 2. Generative QA (Open-ended)
In **Generative (Abstractive) QA**, the model is not limited to quoting the source text. It uses its internal knowledge to write originally phrased answers. While more natural and conversational, this architecture is subject to the "Knowledge Cutoff" of the model's training data. If the model is asked about current events or private corporate data that was not in its pre-training set, it will likely provide an outdated or halluncinated response.

## 3. Knowledge-Based QA
To achieve high-reliability answers, models can be linked to a **Knowledge Graph** or a Structured Database. In this paradigm, a user's natural language question is first converted by the LLM into a formal query (such as **SQL** or **SPARQL**). The result retrieved from the database is then converted back into natural language for the user, ensuring that the answer is anchored in deterministic, verifiable facts.

## 4. Retrieval-Augmented Generation (RAG)
**RAG** is the definitive industrial architecture for QA. It bridges the gap between general intelligence and custom, real-time data.
- **Retriever**: When a question is asked, the system searches a **Vector Database** (Module 02/10) to find the most relevant document chunks.
- **Generator**: These chunks are inserted into the model's prompt as "Grounding Context."
- **The Result**: The model answers the question by "reading" the provided documents, dramatically improving factuality and allowing for the inclusion of citations.

## 5. Evaluation and Benchmark Datasets
QA performance is measured across three axis:
- **Exact Match (EM)**: In extractive tasks, did the model find the precise correct characters?
- **F1-Score**: To what extent did the model's answer overlap with the ground truth?
- **Faithfulness (Metric of RAG)**: Did the generative answer come *only* from the provided documents?
Through specialized datasets like **HotpotQA** (which requires multi-step reasoning across multiple documents), researchers continue to push the boundaries of how deeply a machine can "understand" and query the collective knowledge of humanity.
