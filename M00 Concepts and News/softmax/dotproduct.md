# The Mathematical Framework of Dot Products and Logit Distributions in Large Language Models



## Abstract

Within the architecture of Large Language Models (LLMs) and vector-based retrieval systems, the dot product serves as the primary metric for calculating semantic affinity. This tutorial provides a rigorous examination of how algebraic operations on key-value pairs are transformed into probabilistic distributions through the use of logits and the Softmax function.



---



## 1. Vectorial Representation in Key-Value Paradigms

In high-dimensional latent spaces, semantic information is encoded as vectors within a Key-Value framework. Unlike traditional associative arrays, the "Value" here is a coordinate in $mathbb{R}^d$, where $d$ represents the embedding dimension (e.g., $d=1536$ for OpenAI's `text-embedding-3-small`).



### 1.1 Semantic Mapping

| Entity (Key) | Vector Representation (Value) | Geometric Interpretation |

| :--- | :--- | :--- |

| $text{Canis lupus}$ | $[0.1, 0.9, dots, n]$ | High-ordinality vector in the 'canine' manifold. |

| $text{Lupus}$ | $[0.15, 0.85, dots, n]$ | Proximal vector with high directional alignment. |

| $text{Mensa}$ | $[0.7, 0.1, dots, n]$ | Distal vector in an orthogonal semantic plane. |



---



## 2. Algebraic Foundationalism: The Dot Product

The dot product (Spanish: *Producto Escalar* or *Producto Punto*) is the fundamental operation defining the interaction between a query vector $mathbf{q}$ and a key vector $mathbf{k}$.



### 2.1 Formal Definition

Given two vectors $mathbf{A}, mathbf{B} in mathbb{R}^n$, the dot product is defined as:

$$mathbf{A} cdot mathbf{B} = sum_{i=1}^{n} a_i b_i = |mathbf{A}| |mathbf{B}| cos(theta)$$

where $|mathbf{A}|$ denotes the Euclidean norm and $theta$ represents the subtended angle. In the context of LLMs, this scalar result quantifies the unnormalized projection of one concept onto another.



---



## 3. Metric Divergence: Cosine Similarity vs. Distance

While the dot product is sensitive to vector magnitude, LLMs frequently utilize Cosine Similarity to isolate directional alignment from lexical frequency or document length.



### 3.1 Cosine Similarity (Similitud del Coseno)

By normalizing the dot product by the product of the magnitudes, we derive a value invariant to scale:

$$S_c(mathbf{A}, mathbf{B}) = frac{mathbf{A} cdot mathbf{B}}{|mathbf{A}| |mathbf{B}|}$$

* **Domain:** $[-1, 1]$. In most embedding spaces, values cluster between $[0, 1]$.



### 3.2 Cosine Distance (Distancia del Coseno)

To satisfy the requirements of a formal metric space for optimization algorithms, the Cosine Distance is defined as:

$$

