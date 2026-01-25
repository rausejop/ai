# Chapter 1.2: Fundamentals of Tokenization

## 1. What is Tokenization?
In the architecture of Natural Language Processing (NLP), **Tokenization** is formally defined as the deterministic process of decomposing a continuous sequence of text into a series of discrete, manageable units termed **Tokens**. These units serve as the atomic building blocks of the language model's universe. Without tokenization, a computer perceives text as an undifferentiated stream of bytes; through tokenization, the system gains a structural map of the vocabulary.

## 2. Character vs. Word Tokenization
Historically, the field has navigated a trade-off between two extreme granularities:
- **Character-level Tokenization**: Each individual character (or byte) is a token. This approach eliminates the "Out-of-Vocabulary" (OOV) problem, as every word can be reconstructed. However, it results in extremely long sequences that tax the computational limits of the attention mechanism and lacks the semantic density required for high-level reasoning.
- **Word-level Tokenization**: Documents are split based on known word boundaries. While the tokens are semantically rich, this method necessitates an immense vocabulary and fails when encountering neologisms, typos, or rare grammatical inflections, leading to a high frequency of "Unknown" (`<|unk|>`) tokens.

## 3. Regular Expression Tokenizers
To move beyond simple whitespace splitting, practitioners utilize **Regular Expression (Regex) Tokenizers**. By defining precise patterns—such as `re.split(r'([,.:;?_!"()\']|--|\s)', text)`—developers can ensure that punctuation and structural markers are preserved as independent tokens. This is crucial for capturing the syntactic intent of a sentence (e.g., distinguishing a statement from a query). Modern libraries like `NLTK` and `spaCy` provide pre-built, highly optimized regex engines that handle thousands of linguistic edge cases.

## 4. Issues: Punctuation and Contractions
A significant challenge in rule-based tokenization involves the treatment of **Punctuation** and **Contractions**. Simple splitters often fail to correctly separate a word from an attached comma or full stop. Furthermore, contractions like "don't" or "it's" present a dilemma: should they remain a single unit, or be split into "do n't" and "it 's" to preserve the underlying verb-negation structure? Most modern English tokenizers prefer the latter, as it allows the model to learn the representation of the word "not" consistently across different contractions.

## 5. Whitespace and Rule-Based Tokenization
The earliest and most computationally trivial method is **Whitespace Tokenization**, which assumes that meaning-carrying units are separated by spaces. While effective for English, this method is fundamentally flawed for "unsegmented" languages like Chinese or Japanese. Consequently, the field has moved toward **Rule-Based** systems that integrate language-specific heuristics and statistical patterns to define boundaries. Through these mechanisms, tokenization provides the necessary scaffolding for the model to begin its transition from symbols to meaningful latent representations.
