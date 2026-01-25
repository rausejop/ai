# Chapter 3.2: The Transformer: Attention Is All You Need

## 1. The Transformer Block Structure
The **Transformer Block** is the fundamental atomic unit of modern Large Language Models. Unlike older recurrent or convolutional blocks, a Transformer block consists of two primary sub-layers: a **Multi-Head Self-Attention** mechanism followed by a **Point-wise Feed-Forward Network**. These sub-layers are interconnected via residual connections and stabilized by layer normalization, creating a "deep highway" that allows gradients to flow through hundreds of layers without degradation.

## 2. Self-Attention Mechanism Explained
At the heart of the block lies the **Scaled Dot-Product Attention**. This mechanism allows every token in a sequence to "attend" (assign weight) to every other token. For setiap input embedding, the model computes three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**.
- **The Process**: The model calculates the compatibility between a Query and all Keys using a dot product, divides by the square root of the dimension ($\sqrt{d_k}$) for numerical stability, and applies a softmax to create a probability distribution.
- **The Result**: A context-aware vector that is a weighted sum of the Values, effectively capturing the semantic essence of the token's environment.

## 3. Multi-Head Attention and Why It Works
To capture the complexity of human language, the model utilizes **Multi-Head Attention**. Instead of a single attention function, it performs multiple attention operations in parallel (heads).
- **The Logic**: One head may focus on grammatical dependencies (e.g., subject-verb pairing), while another captures coreference (e.g., resolving a pronoun). By concatenating and projecting the outputs of these heads, the model develops a rich, multi-dimensional understanding of the text.

## 4. Positional Encodings: The Need for Position
Because Transformers process all tokens in parallel, they are "permutation invariant"—they have no inherent sense of word order. To resolve this, **Positional Encodings** are added to the input embeddings. Using fixed sinusoidal functions or learned vectors, these encodings provide each token with a unique "temporal coordinate," allowing the model to distinguish between "The dog bit the man" and "The man bit the dog."

## 5. Feed-Forward Network and Layer Normalization
Following the attention phase, each token vector is passed independently through a **Position-wise Feed-Forward Network (FFN)**. This layer typically performs a non-linear expansion (increasing dimensionality) followed by a contraction, allowing the model to store and process complex logical features. **Layer Normalization** is applied before each sub-layer to re-scale the activations, ensuring that the mathematical signals remain stable during the thousands of optimization steps required for Large Language Model training.
