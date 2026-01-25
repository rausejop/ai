# Chapter 2.1: Introduction to Embeddings: Why Vectors?

## 1. What are Embeddings?
In the landscape of modern Artificial Intelligence, **Embeddings** are defined as the mapping of discrete categorical variables (such as words, tokens, or entities) into a continuous, high-dimensional vector space. Unlike symbolic representations that treat words as isolated markers, embeddings provide a mathematical framework where linguistic concepts are represented as points in a geometric manifold. This dimensionality reduction—transforming a vocabulary of millions of words into a dense vector of 768 or 1536 dimensions—is what enables neural networks to process human language with unprecedented efficiency.

## 2. The Distributional Hypothesis
The theoretical foundation of embeddings is anchored in the **Distributional Hypothesis**, famously articulated by J.R. Firth in 1957: *"You shall know a word by the company it keeps."* In computational terms, this implies that words occurring in similar linguistic environments—that is, surrounded by similar neighbor words—likely share similar semantic properties. By analyzing trillions of contexts, a model can "learn" information about a concept without ever being shown an explicit definition.

## 3. From One-Hot to Dense Vectors
Historically, text was represented via **One-Hot Encoding**, where a word was a sparse vector with a single "1" at its index and "0" elsewhere. This method has two fatal flaws:
- **Dimensionality**: In a large corpus, one-hot vectors become prohibitively long and sparse.
- **Semantic Orthogonality**: In one-hot space, all vectors are orthogonal, meaning the dot product of "Cat" and "Kitten" is zero, implying no similarity.
**Dense Vectors** (Embeddings) resolve this by projecting words into a shared, continuous space. In this latent space, semantically related tokens are positioned closer together, allowing the model to apply the full power of linear algebra to linguistic reasoning.

## 4. Measuring Vector Similarity (Cosine Similarity)
Once words are represented as vectors, we require a metric to quantify their similarity. The industry standard is **Cosine Similarity**, which measures the cosine of the angle $\theta$ between two vectors $A$ and $B$:
$$\text{sim}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}$$
Unlike Euclidean distance, which can be distorted by document length or frequency, Cosine Similarity focuses on the **Orientation** of the vectors. This ensures that "Dog" and "Puppy" are recognized as semantically identical regardless of how often they appear in a given text. Through these integrated mechanisms, embeddings provide the "numerical dialect" through which machines perceive the nuances of human thought.
