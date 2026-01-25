# Chapter 2.1: Introduction to Embeddings: Why Vectors?

## The Philosophical and Mathematical Foundations

The transition from symbolic logic to vector-based representation marks one of the most profound shifts in the history of Artificial Intelligence. At its core, the concept of embeddings is anchored in the **Distributional Hypothesis**, famously articulated by J.R. Firth in 1957: *"You shall know a word by the company it keeps."* In the context of computational linguistics, this hypothesis implies that words occurring in similar linguistic environments—that is, surrounded by similar sets of neighbor words—likely share similar semantic properties.

Historically, computers represented words as discrete symbols. In a **One-Hot Encoding** scheme, each word in a vocabulary of size $V$ is represented as a sparse vector where only one dimension is "hot" (assigned a value of 1) and all others are "cold" (0). This symbolic approach, while intuitive, suffers from three critical technical flaws. First, it is prohibitively high-dimensional for modern corpora. Second, it is discrete, providing no mechanism for partial similarity. Third, and most importantly, it is **Orthogonal**; mathematically, the dot product of any two distinct one-hot vectors is zero, implying that "cat" is as distant from "kitten" as it is from "astrophysics."

## The Emergence of Dense Representations

To overcome these limitations, modern NLP utilizes **Dense Embeddings**. Instead of a sparse vector of size $V$, each word is mapped to a continuous vector of a fixed, relatively small dimension $d$ (typically 128, 768, or 1536). These vectors occupy a continuous latent space where semantic similarity is captured through geometric proximity.

In deep learning architectures, such as those analyzed by Sebastian Raschka, the transition to vectors is mediated by an **Embedding Layer**. Technically, this layer is a weight matrix $W \in \mathbb{R}^{V \times d}$ that serves as a high-speed look-up table. Multiplying a one-hot vector $x_i$ by this matrix is equivalent to extracting the $i$-th row of $W$. During the training of a Large Language Model (LLM), these weights are not static; they are parameters that the model optimizes to ensure that semantically related tokens are positioned closer together in the hyperspace.

## Measuring Meaning in Latent Space

Once words are represented as vectors, the model requires a mathematical metric to evaluate "meaning distance." The industry standard is **Cosine Similarity**, defined as:
$$\text{sim}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}$$
Unlike Euclidean distance, which measures the straight-line distance between points, Cosine Similarity focuses on the angle between vectors. This makes it invariant to the magnitude (length) of the vectors, ensuring that a word used frequently in a long document and the same word used in a short one are recognized as semantically identical.

By translating human language into this numerical dialect, we enable the model to apply the full power of linear algebra and calculus to linguistic reasoning. The result is a system that "understands" that words are not just labels, but points in a complex, multi-dimensional map of human thought.
