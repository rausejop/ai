# Chapter 2.2: Word2Vec: CBOW and Skip-gram

## The 2013 Breakthrough: Learning by Prediction

The introduction of **Word2Vec** by Mikolov et al. at Google represented a pivotal moment in NLP, demonstrating for the first time that high-quality, dense embeddings could be learned efficiently from vast amounts of unlabelled text. Word2Vec moved the field away from expensive hand-curated semantic nets toward a self-supervised learning paradigm where the text itself serves as the teacher. The framework consists of two primary neural architectures: **Continuous Bag-of-Words (CBOW)** and **Skip-gram**.

### Continuous Bag-of-Words (CBOW)

The CBOW architecture is designed with the objective of predicting a **target word** based on its surrounding **context words**. Conceptually, if the model is given the context "The cat [ ] on the mat," its goal is to correctly identify the missing word "sat." Technically, CBOW averages or sums the embedding vectors of the context words into a single representation. This aggregated vector is then passed through a projection layer and a softmax output layer to predict the probability distribution of the center word. While computationally faster and highly effective for frequent words, CBOW effectively "smears" the context, which can sometimes lead to a loss of fine-grained semantic detail.

### Skip-gram: High-Resolution Contextualization

The Skip-gram architecture invert the logic of CBOW. Its objective is to predict the **surrounding context words** given a single **target word**. If the input is "cat," the model attempts to predict the likelihood of "the," "sat," and "on" appearing nearby. As noted in the works of Raschka, Skip-gram is often the preferred choice for modern researchers. Because it forces the model to learn multiple predictions for a single input, it is significantly better at capturing stable representations for rare words and nuances that might be lost in the averaging process of CBOW.

### Scaling and Optimization: Negative Sampling

A primary technical challenge in training Word2Vec is the **Softmax Bottleneck**. Calculating the probability of a word out of a vocabulary of one million units requires a massive summation in the denominator of the softmax function at every training step. Word2Vec resolves this through **Negative Sampling (NS)**. Instead of a full classification task, the problem is reframed as a binary classification: "Is this context word a real neighbor of the target word (Positive), or is it a random 'noise' word from the vocabulary (Negative)?" By only updating the weights for the positive sample and a handful of negative distractors, the computational complexity is reduced by several orders of magnitude.

### Linear Analogies and Semantic Arithmetic

The most famous technical achievement of Word2Vec is the discovery that its learned vectors satisfy linear semantic relationships. This suggests that the model has internally organized the latent space into meaningful dimensions such as gender, verb tense, and geographic hierarchy. Mathematically, it was shown that:
$$\text{vec("King") } - \text{ vec("Man") } + \text{ vec("Woman") } \approx \text{ vec("Queen")}$$
This discovery proved that deep learning could capture abstract concept logic without any explicit linguistic rules, providing the first glimpse into the "reasoning" capabilities that would eventually evolve into the Large Language Models of today.
