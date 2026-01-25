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

## 📊 Visual Resources and Diagrams

- **The SQuAD Span-Prediction Flow**: A diagram showing how BERT identifies the Start/End indices for an answer.
    - [Source: Jay Alammar - The Illustrated BERT (Question Answering section)](https://jalammar.github.io/images/bert-squad.png)
- **The RAG Architectural Blueprint**: An end-to-end infographic by Meta AI showing the retriever-generator interplay.
    - [Source: Lewis et al. (2020) - Retrieval-Augmented Generation (Fig 1)](https://arxiv.org/pdf/2005.11401.pdf)

## 🐍 Technical Implementation (Python 3.14.2)

Low-latency **Extractive Question Answering** using the **RoBERTa-Base** engine on Windows.

```python
from transformers import pipeline

def corporate_faq_resolver(question: str, context: str):
    """
    Performs high-precision extractive QA on a target context.
    Ideal for technical manual search.
    Compatible with Python 3.14.2.
    """
    # 1. Initialize the QA pipeline
    qa_engine = pipeline(
        "question-answering", 
        model="deepset/roberta-base-squad2"
    )
    
    # 2. Inference pass
    result = qa_engine(question=question, context=context)
    
    return result

if __name__ == "__main__":
    manual_context = """
    The reactor's core must be maintained below 350 degrees Celsius. 
    If the temperature exceeds this threshold, the automated nitrogen 
    cooling system will engage. Emergency manual override is located 
    at Terminal 4B.
    """
    
    q1 = "What happens if the temperature goes over 350 degrees?"
    res = corporate_faq_resolver(q1, manual_context)
    
    print(f"Question: {q1}")
    print(f"Answer: {res['answer']}")
    print(f"Confidence: {res['score']:.4f}")
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Rajpurkar et al. (2016)**: *"SQuAD: 100,000+ Questions for Machine Comprehension of Text"*. The paper that defined modern QA.
    - [Link to ArXiv](https://arxiv.org/abs/1606.05250)
- **Lewis et al. (2020)**: *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. The origin of RAG.
    - [Link to ArXiv](https://arxiv.org/abs/2005.11401)

### Frontier News and Updates (2025-2026)
- **Microsoft Research (January 2026)**: Release of *Graph-Reranker-V2*, which improves QA accuracy by 40% by checking the retrieved text against a Knowledge Graph.
- **OpenAI News**: Discussion on "System-2 Reasoning"—how o-series models double-check their own answers to ensure 99.9% exact match in clinical QA.
- **TII Falcon Insights**: Announcement of the *Falcon-QA-Atlas*, the first multimodal QA model that can answer questions about images, spreadsheets, and videos simultaneously.
