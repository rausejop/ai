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

## 📊 Visual Resources and Diagrams

- **The IOB Tagging Schema Visualization**: A clear mapping of tokens to B-I-O labels.
    ![The IOB Tagging Schema Visualization](https://jalammar.github.io/images/bert-conll-ner.png)
    - [Source: Jay Alammar - The Illustrated BERT (NER Section)](https://jalammar.github.io/images/bert-conll-ner.png)
- **CRF Probabilistic Graph**: A visualization of the transition probabilities between adjacent entity labels.
    ![CRF Probabilistic Graph](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Conditional_random_field.png/440px-Conditional_random_field.png)
    - [Source: Wikipedia - Conditional Random Field](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Conditional_random_field.png/440px-Conditional_random_field.png)

## 🐍 Technical Implementation (Python 3.14.2)

Low-level IOB Tagging and classification using `spaCy` 4.x on Windows.

```python
import spacy # Importing the main spaCy NLP library for industrial-scale text processing
from spacy.tokens import Doc # Importing the Doc class to handle structured tokenized representations

def custom_iob_analyzer(text: str): # Defining a function to extract and explain IOB entity tags
    """ # Start of the function's docstring
    Extracts raw IOB tags for diagnostic analysis of entity boundaries. # Explaining the pedagogical goal of IOB analysis
    Compatible with Python 3.14.2 and spaCy 4.0. # Specifying the target version for current Windows-based installations
    """ # End of docstring
    nlp = spacy.load("en_core_web_md") # Loading the mid-sized English transformer-based model for feature extraction
    doc = nlp(text) # Executing the full NLP pipeline to perform tokenization, tagging, and NER
    
    iob_report = [] # Initializing an empty list to store the resulting token-level diagnostic data
    for token in doc: # Iterating through varje token in the processed document
        # ent_iob_: B, I, or O # Pedagogical note identifying the raw IOB status code
        # ent_type_: The entity label (ORG, PERSON, etc.) # Pedagogical note identifying the categorical assignment
        # Constructing the combined IOB-Label string (e.g., 'B-ORG')
        tag_string = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O" # Logic for structured tag creation
        iob_report.append({ # Appending the token metadata to our result report
            "token": token.text, # Storing the original verbatim token string
            "tag": tag_string # Storing the final formatted IOB tag
        }) # Closing the dictionary append
        
    return iob_report # Returning the list of token-label pairs for console display

if __name__ == "__main__": # Entry point check for the standalone execution script
    # Defining a sample sentence with multi-word and geographic entities for diagnostic verification
    sample = "The London Stock Exchange is located in the UK." 
    report = custom_iob_analyzer(sample) # Executing the custom IOB analysis on the test sample
    
    print(f"Input: {sample}\n") # Displaying the original text for easy comparison
    # Iterating through each row in the generated IOB report for tabular display
    for r in report: # Iterating through the result set
        print(f"{r['token']:<15} | {r['tag']}") # Displaying the token and its associated IOB tag in a formatted table
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Lafferty et al. (2001)**: *"Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"*. The breakthrough paper for sequence tagging.
    - [Link to ICML Archive](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
- **Collobert et al. (2011)**: *"Natural Language Processing (Almost) from Scratch"*. Introducing neural NER architectures.
    - [Link to JMLR](https://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

### Frontier News and Updates (2025-2026)
- **Google DeepMind (Late 2025)**: Development of *Vision-NER*—identifying entities directly in images and PDF layouts without performing OCR first.
- **NVIDIA AI Blog**: "Million-Entity NER"—Using H200 clusters to extract entities from entire library archives in seconds.
- **Microsoft Research 2026**: Announcement of *Logic-NER*, an engine that uses LLM reasoning to ensure that extracted entities satisfy logical constraints (e.g., avoiding an entity being a "Person" and a "Location" simultaneously).
