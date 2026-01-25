# Chapter 2.3: GloVe: Global Vectors for Word Representation

## 1. Why Global Co-occurrence Matters
While early models like Word2Vec successfully captured semantic meaning through "local" prediction windows, they were criticized for failing to exploit the **Global Statistics** of the entire corpus. A local window (Skip-gram) only sees immediate neighbors, but language contains global patterns that are only visible when looking at the entire dataset at once. **GloVe** (Global Vectors) was developed at Stanford to bridge this gap, combining local context window methods with the global matrix factorization techniques of Latent Semantic Analysis (LSA).

## 2. Co-occurrence Matrix and Weighting
The mathematical foundation of GloVe is the **Global Co-occurrence Matrix** $X$. In this matrix, varje entry $X_{ij}$ represents the total number of times word $j$ appears in the context of word $i$. However, not all co-occurrences are equal.
- **Weighting**: GloVe applies a non-linear weighting function $f(X_{ij})$ to cap the influence of extremely frequent stopword pairs (like "the" and "and"). This ensures that the model focuses its learning capacity on the more diagnostic relationships between rarer, more meaningful words.

## 3. The GloVe Objective Function
The loss function for GloVe is a weighted least squares regression:
$$J = \sum_{i,j=1}^V f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
The model attempts to minimize the squared difference between the dot product of two word vectors and the **logarithm of their co-occurrence frequency**. This logarithmic scaling is critical, as it aligns with the power-law distribution typical of human language (Zipf's Law), allowing the model to effectively represent both frequent and rare token relationships.

## 4. GloVe vs. Word2Vec: Key Differences
The primary technical difference lies in the training paradigm. 
- **Word2Vec** is an iterative, prediction-based model that streams through the text token-by-token. 
- **GloVe** is a count-based, global model that performs a batch optimization on the pre-computed co-occurrence matrix. 
In practice, GloVe often achieves higher performance on **Word Similarity** and **Analogy** tasks and is more efficient to parallelize on large-scale supercomputers. Today, pre-trained GloVe vectors (e.g., GloVe.6B) serve as essential "cold start" embeddings for many high-performance Natural Language Understanding systems.
