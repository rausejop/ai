# Chapter 6.4: Knowledge-Augmented Text Enrichment

## 1. What is a Knowledge Graph? (Triples and Relations)
A **Knowledge Graph (KG)** represents information not as a string of text, but as a formal network of **Entities** (Nodes) and their **Relationships** (Edges).
- **The Triple**: The primary unit is the subject-predicate-object triple, e.g., `(Paris, is_capital_of, France)`. 
- **KGs**: Wikidata serves as the definitive open source graph, while organizations build proprietary graphs to track internal facts (e.g., `(Project A, managed_by, Employee B)`). KGs provide the deterministic "Anchor of Truth" that neural networks lack.

## 2. KG Embedding Techniques (e.g., TransE)
To bridge the gap between a "node" in a graph and a "vector" in an LLM, we use **KG Embeddings**.
- **TransE (Translational Embeddings)**: An algorithm that maps entities and relations into the same latent space such that if a triple $(h, r, t)$ exists, the vectors follow the rule $h + r \approx t$. This allows the model to perform "Link Prediction"—mathematically inferring missing facts by calculating vector arithmetic between known nodes.

## 3. Knowledge Graph Completion and Text Enrichment
Knowledge-augmented NLP uses these graphs to **Enrich** the model's understanding. When an LLM processes the mention of "Ada Lovelace," a background process can retrieve her KG neighbors ("Charles Babbage," "Analytical Engine," "First Programmer"). This extra "Technical Context" prevents the model from hallucinating and allows it to achieve a deeper reasoning capability than a model trained on text alone.

## 4. Retrieval-Augmented Generation (RAG) with KGs
The most advanced form of RAG involves **Graph-Retrieval**. Instead of retrieving simple text chunks (Module 05), the system retrieves a **Sub-graph** of related facts. This provides the LLM with a rigid structure of verified information, which it then uses to write a natural language explanation. This "Graph-to-Text" pipeline is the current gold standard for building explainable and fact-checked AI in fields like medicine and automated law.

## 5. Use Cases: Explainability and Fact Verification
The primary utility of KG integration is **Reliability**. Because a KG is human-verified, it can be used to "Audit" the LLM's output. If a model generates a false claim about a corporate hierarchy, a background process can check the KG, find the correct triple, and trigger an immediate correction with a citation. Through this grounding cycle, Knowledge Graphs transform probabilistic "Stochastic Parrots" into reliable, deterministic intelligence agents.
