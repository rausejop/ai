# Chapter 4.4: NEL: Mapping to Knowledge Bases

## 1. Connecting Text to the Real World
**Named Entity Linking (NEL)** represents the terminal objective of the entity-centric pipeline: anchoring a linguistic mention to a specific, unique entry in a global **Knowledge Base (KB)**. While NER identifies the "what" and NED resolves the "which," NEL handles the entire engineering architecture required to link human discourse to the structured repositories of human knowledge.

## 2. What is Named Entity Linking (NEL)?
NEL is the process of mapping a detected entity span to a **Unique Identifier** (e.g., a Wikidata ID like `Q9333` for New York City). This ensures that no matter how many synonyms or aliases a user provides ("NYC", "New York", "The Big Apple"), the system always points to the same mathematical node in your data architecture.

## 3. Role of Knowledge Bases (KBs) (e.g., Wikipedia, Wikidata)
The technical foundation of NEL is the external KB:
- **Wikipedia**: Provides the primary source for textual descriptive metadata and entity summaries.
- **Wikidata / DBpedia**: Large-scale structured databases where information is stored as **Triples** (Subject $\rightarrow$ Predicate $\rightarrow$ Object). These systems provide the "World Knowledge" that allows an AI to understand the relationships between people, places, and events.

## 4. NEL Pipeline: Mention Detection, Candidate Generation, Ranking
A production-grade NEL system operates through a sequential pipeline:
1.  **Mention Detection (NER)**: Identifying the string to be linked.
2.  **Candidate Generation**: Using fuzzy string matching and alias tables to find all potential KB matches. High **Recall** is critical at this stage to ensure the correct entity is included in the list.
3.  **Candidate Ranking (NED)**: Using deep learning models to score each candidate based on semantic similarity and document coherence.
4.  **NIL Detection**: Identifying if a mention refers to an entity that **does not exist** in the KB (e.g., a person mentioned in a private email who is not a public figure).

## 5. NEL vs. NED: A Clear Comparison
While the terms are often used interchangeably, they represent different technical focuses:
- **NED (Disambiguation)** is a **Decision Process**: choosing between candidates (e.g., "Which Washington?").
- **NEL (Linking)** is a **System Architecture**: the complete end-to-end framework that includes detection, search, ranking, and final grounding in a persistent dataset. Together, they provide the "Universal Identification Layer" that allows AI to behave as a reliable, factually-grounded cognitive agent.
