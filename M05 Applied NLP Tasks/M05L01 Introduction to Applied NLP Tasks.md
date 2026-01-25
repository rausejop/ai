# Chapter 5.1: Introduction to Applied NLP Tasks

## The Convergence of Theory and Production

Having rigorously dissected the internal mechanics of the Transformer architecture, the geometry of latent vector spaces, and the infrastructure of entity grounding, this textbook now transitions to the primary objective of the field: the application of these technologies to solve high-value functional problems. **Applied NLP** is the domain where mathematical probability is transformed into functional utility.

### Functional Categorization: Discriminative vs. Generative

In the current landscape, applied tasks are broadly bifurcated into two technical paradigms:

1.  **Discriminative Tasks**: These involve the categorization or labeling of input text. The model's objective is to reduce the input into a set of discrete, predefined labels (e.g., "Is this email spam?", "What is the sentiment of this review?"). These tasks primarily utilize **Encoder-only** or **Encoder-Decoder** architectures and are measured through rigid statistical metrics like F1-Score and Accuracy.
2.  **Generative Tasks**: These involve the production of new, fluent text based on a given prompt or document (e.g., summarizing an article, translating a sentence, or generating a legal clause). These tasks rely on **Decoder-only** or **Encoder-Decoder** architectures and require more nuanced evaluation frameworks that account for linguistic diversity and factual faithfulness.

### The Benchmarking Culture: GLUE and Beyond

The progress of applied NLP is governed by standardized benchmarks that provide a unified metric for model comparison. The **GLUE (General Language Understanding Evaluation)** and its successor **SuperGLUE** represent the industry record. These benchmarks encompass tasks ranging from natural language inference (detecting if one sentence implies another) to logical entailment and coreference resolution. Success on these benchmarks is the primary signal that a model has achieved a level of general "Reasoning," which can then be fine-tuned for specialized industrial domains.

### Error Analysis and the Challenge of Context

Deploying NLP in production requires a deep understanding of **Failure Modes**. Unlike simple code, NLP models are probabilistic and "opaque."
- **Data Drift**: A model trained on 2020 news data may fail to correctly classify 2024 political discourse because the underlying language and sentiment have shifted.
- **Hallucination vs. Factuality**: Especially in generative tasks, a model might produce a linguistically perfect sentence that is factually bankrupt. 
- **Modular vs. End-to-End**: A critical design decision for practitioners. While using a single Large Language Model (End-to-End) is simpler to implement, a **Modular Pipeline** (e.g., specialized models for NER followed by a classifier) is often more explainable, controllable, and cost-effective for high-volume enterprise operations. As we explore the specific tasks in this module, we will keep these production-level trade-offs at the center of our technical analysis.
