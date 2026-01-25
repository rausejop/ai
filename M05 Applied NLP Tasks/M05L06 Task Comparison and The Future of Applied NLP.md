# Chapter 5.6: Task Comparison and The Future of Applied NLP

## 1. Comparing Discriminative vs. Generative Tasks
The modern NLP landscape is a spectrum of logic. **Discriminative Tasks** (Classification, Sentiment, NER) focus on precision and labeling; they are the "Evaluators" of AI. **Generative Tasks** (Summarization, QA, Code Generation) focus on synthesis and creation; they are the "Builders." In a production environment, success requires a strategic balance: using discriminative models to verify and filter the outputs of generative ones.

## 2. Task Interdependencies (e.g., Classification supporting QA)
Applied NLP is rarely a single operation. Modern systems are built as **Task Orchestrations**. For instance:
1.  **Classification** identifies the user's intent (e.g., "This is a billing question").
2.  **NER/NEL** extracts the specific account identifiers.
3.  **RAG** retrieves the relevant billing documentation.
4.  **Summarization** condenses the finding into a brief, polite response.
By treating individual tasks as modular components, we build robust intelligent agents capable of solving complex multi-step problems.

## 3. Ethical Considerations in Applied NLP
As LLMs are deployed into high-stakes industries (Law, Finance, Medicine), we must address fundamental ethical risks:
- **Bias**: Ensuring classification models do not develop prejudices based on patterns in the training data.
- **Privacy**: Preventing the model from "remembering" or leaking PII (Personally Identifiable Information) from its training sets.
- **Transparency**: The technical requirement that AI responses must be explainable and traceable to a source document, especially in regulated environments.

## 4. The Unified View of LLMs
The ultimate trajectory of the field is toward **The Foundation Model**. We are moving away from having 10 separate models for 10 separate tasks. Instead, we use a single, massive Large Language Model that is "prompted" or "fine-tuned" to perform any applied task on-demand. This unified approach simplifies infrastructure and allows for the emergence of "Cross-Task Reasoning," where the model's knowledge of summarization helps it be better at question answering, creating a virtuous cycle of artificial intelligence.
