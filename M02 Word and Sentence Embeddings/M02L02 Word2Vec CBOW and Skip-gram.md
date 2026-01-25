# Chapter 2.2: Word2Vec: CBOW and Skip-gram

## 1. Word2Vec Architecture Overview
Introduced in 2013 by Mikolov et al. at Google, **Word2Vec** represented a paradigm shift in Natural Language Processing. It moved the field away from expensive, hand-curated semantic nets toward a self-supervised learning paradigm where the text itself serves as the teacher. Word2Vec is characterized by its use of a shallow, two-layer neural network designed to produce dense word embeddings by predicting words based on their context.

## 2. Continuous Bag-of-Words (CBOW) Model
The **CBOW** model is architected to predict a **target word** given a set of surrounding **context words**. 
- **Mechanism**: The model takes the embeddings of the context words (e.g., "The cat [ ] on the mat"), averages or sums them into a single aggregate vector, and then attempts to identify the most probable center word ("sat"). 
- **Performance**: CBOW is computationally efficient and performs exceptionally well on frequent words, though it can sometimes struggle with capturing rare words due to the "smearing" effect of the context averaging.

## 3. Skip-gram Model Explained
The **Skip-gram** architecture inverts the logic of CBOW. Its objective is to predict the **surrounding context words** given a single **target word**.
- **Mechanism**: If the input is "sat," the model attempts to predict the likelihood of "cat," "on," and "mat" appearing nearby.
- **Advantage**: As noted by Raschka, Skip-gram is often the preferred choice for research. Because it forces the model to learn multiple predictions for a single input, it is significantly better at capturing stable representations for rare words and fine-grained semantic nuances.

## 4. Negative Sampling and Optimization
Training a neural network to predict a word out of a vocabulary of one million is computationally expensive due to the massive softmax operation required at the output layer. Word2Vec resolves this through **Negative Sampling (NS)**.
- **Optimization**: Instead of a full classification task, the model reframes the problem: "Is this context word a real neighbor (Positive), or a random 'noise' word (Negative)?" By only updating the weights for the positive sample and a handful of negative distractors, the training speed increases by several orders of magnitude.

## 5. Demonstrating Word Analogies (e.g., King - Man + Woman = Queen)
The most profound technical achievement of Word2Vec is the discovery that its learned vectors satisfy linear semantic relationships. This implies that the model has internally organized the latent space into meaningful dimensions (gender, tense, geography). Mathematically, it was shown that:
$$\text{vec("King") } - \text{ vec("Man") } + \text{ vec("Woman") } \approx \text{ vec("Queen")}$$
This discovery proved that emergent "logic" could be captured through raw statistical association, laying the foundation for the complex reasoning of modern transformers.
