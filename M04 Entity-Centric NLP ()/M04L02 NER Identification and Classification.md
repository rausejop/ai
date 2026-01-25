# Chapter 4.2: NER: Identification and Classification

## 1. What is Named Entity Recognition?
**Named Entity Recognition (NER)** is formally defined as a **Sequence Tagging** task. Given a sequence of tokens $S = [t_1, t_2, \dots, t_n]$, the objective is to assign a label $y_i$ to varje token where $y_i$ indicates both the presence of an entity and its semantic category. This process transforms a noisy string of words into a structured data format, providing the "Who," "Where," and "What" that drives high-stakes reasoning in enterprise AI.

## 2. Standard Entity Types (PER, ORG, LOC)
While modern systems can be trained for any domain, the industry standards are built upon universal core categories:
- **PER (Person)**: Identifying individuals (e.g., "Marie Curie").
- **ORG (Organization)**: Detecting corporate, political, or social bodies (e.g., "NASA").
- **LOC (Location)**: Identifying geographic points, cities, or countries (e.g., "Tokio").
Advanced models extend these to include **GPE** (Geopolitical Entities), **DATE**, **MONEY**, and specialized identifiers like **ISIN** codes in finance or **Gene Sequences** in biology.

## 3. Sequence Tagging Methods (CRF vs. Bi-LSTM/Transformer)
Historically, NER was solved using **Conditional Random Fields (CRFs)**—discriminative probabilistic models that capture the dependency between adjacent labels. In the deep learning era, this was improved with **Bi-LSTM-CRF** architectures. Today, **Transformer-based NER** (using models like BERT or RoBERTa) has set new benchmarks. The self-attention mechanism allows the model to use distant context (e.g., a verb occurring ten tokens later) to correctly identify if a name refers to a person or a company name, achieving near-human precision.

## 4. Annotation Formats (IOB Tagging)
To represent multi-word entities (like "The United Nations"), practitioners use structured tagging schemes, the most common being **IOB (Inside-Outside-Beginning)**:
- **B (Beginning)**: Marks the first token of an entity span.
- **I (Inside)**: Marks subsequent tokens within the same span.
- **O (Outside)**: Indicates the token belongs to no recognized entity.
**Example**: In "Headquarters in Brussels," the tags would be `Headquarters (O) in (O) Brussels (B-LOC)`. This deterministic format ensures that spans can be extracted without boundary ambiguity.

## 5. Evaluation Metrics for NER
Evaluating NER requires a more rigorous approach than standard classification. Because the exact "start" and "end" of the span are as important as the label, researchers use **Span-based F1-Score**. A prediction is only marked as a "True Positive" if both the boundary and the category are perfectly correct. Failing to include the word "International" in "Amnesty International" would count as an error, ensuring that the resulting data structures are reliable for downstream automated workflows.
