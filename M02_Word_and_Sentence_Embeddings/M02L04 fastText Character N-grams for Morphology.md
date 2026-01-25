# Chapter 2.4: fastText: Character N-grams for Morphology

## Looking Inside the Token

The primary limitation of models such as Word2Vec and GloVe is their treatment of words as "atomic units." In these architectures, each word has a unique vector, but there is no structural connection between shared roots or morphological variations. For example, "learn," "learning," and "learned" are treated as three entirely unrelated symbols. **fastText**, developed by Bojanowski et al. at Facebook AI Research (FAIR), addresses this through a sub-lexical approach using **Character N-grams**.

### The Mechanics of Subword Embeddings

fastText decomposes each word into a set of character n-grams. For instance, given the word "apple" and $n=3$, the model generates the following sub-units:
- `<ap`, `app`, `ppl`, `ple`, `le>`
(Note the use of `<` and `>` to mark the start and end of the word).

The final vector for a word is not just a direct look-up; it is the **Centroid (Average)** of the vector for the full word and the vectors for all its constituent character n-grams. Technically, this allows the model to share parameters across words that look similar.

### Solutions to the Out-of-Vocabulary (OOV) Problem

The most significant technical advantage of fastText is its handling of **Out-of-Vocabulary** words. In traditional models, encountering a word that was not in the training set leads to an "Unknown" token. In fastText, however, even if the model has never seen the word "biotransformation," it can construct a highly accurate embedding by summing the vectors for its constituent n-grams like `bio`, `trans`, and `form`.

This capability makes fastText disproportionately effective for morphologically rich languages (e.g., Arabic, Finnish, or Turkish) where words take numerous forms that may appear only once or twice in a corpus. It also provides inherent robustness against **Typographical Errors**; since a typo like "proccessor" shares many n-grams with "processor," the resulting embedding remains semantically useful.

### Efficiency and Classifier Integration

Beyond its role as an embedding generator, fastText is also a high-performance library for **Text Classification**. By utilizing a hierarchical softmax and an optimized linear model, it can train on billions of tokens in minutes. While it lacks the deep contextual understanding of later Transformer models, its speed and OOV resilience make it the industry standard for large-scale, real-time classification tasks where low latency is a primary requirement. Despite these strengths, fastText remains a **Static Embedding** model; it cannot distinguish between the multiple meanings of a polysemous word based on the sentence it appears in—a problem that necessitated the birth of modern Transformers.
