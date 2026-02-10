# A Historical and Functional Analysis of the Softmax Operator



## 1. Historical Lineage and Origins

The mathematical foundations of the Softmax function—formally known as the \*\*multinomial logit\*\* or \*\*Gibbs distribution\*\*—predate modern computational science. Its conceptual origin lies in 19th-century statistical mechanics, specifically the work of \*\*Ludwig Boltzmann (1868)\*\*, who utilised the exponential form to describe the probability distribution of particles across discrete energy states.



In the twentieth century, the function was adapted into mathematical statistics for multinomial regression. However, its formal introduction to the field of connectionism and artificial neural networks is largely attributed to \*\*John S. Bridle (1989)\*\*. Bridle proposed the Softmax nonlinearity as a means to ensure that the outputs of a feedforward network represent a valid probability distribution, thereby facilitating a principled statistical interpretation of classification tasks.



### 1.1. The Roots: Ludwig Boltzmann (1868)

The fundamental concept of Softmax originates from the \*\*Boltzmann distribution\*\* (or Gibbs distribution). In the 19th century, the Austrian physicist Ludwig Boltzmann utilised this exponential form to describe the probability of a system being in a specific state of energy. 



The formula employed today to classify digital patterns (such as distinguishing between images of cats and dogs) is structurally identical to the one describing the behaviour of gas molecules:

$$P(i) = \\frac{e^{-E\_i / kT}}{\\sum\_{j} e^{-E\_j / kT}}$$



### 1.2. The Statistical Bridge: R.A. Fisher and Multinomial Models

In the early 20th century, statisticians such as \*\*Ronald Fisher\*\* began implementing similar functions for multinomial logistic regression models. In this context, the objective was to find a robust method to normalise a vector of real numbers into a probability distribution where the sum of all elements equals 1 (100%).



### 1.3. The Leap to Artificial Intelligence (1980s – 1990s)

The term "Softmax" gained significant traction within the neural network community during the late 1980s. A pivotal milestone was reached by:

\* \*\*John S. Bridle (1989)\*\*: Bridle was among the first to formally propose the use of this function in neural networks for pattern classification. He provided the nomenclature we use today in his seminal paper, \*"Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition"\*.





## 2. Etymology: Why 'Softmax'?

The nomenclature is a portmanteau reflecting its functional characteristics:

\* \*\*Max\*\*: The operator mimics the 'Maximum' function by magnifying the largest input value (logit) relative to the others.

\* \*\*Soft\*\*: Unlike the 'Hardmax' or 'Argmax' functions—which are non-differentiable and binary (assigning 1 to the maximum and 0 elsewhere)—Softmax provides a continuous, 'soft' approximation. This differentiability is crucial for gradient-based optimisation, as it allows the backpropagation algorithm to adjust weights through a smooth probability landscape.



## 3. Application in Large Language Models (LLMs)

In the context of contemporary Transformer-based architectures, Softmax is an indispensable component employed in two primary stages:



### 3.1. Scaled Dot-Product Attention

Within the self-attention mechanism, Softmax is applied to the scores derived from the interaction between Queries ($Q$) and Keys ($K$). It normalises these scores into a distribution of 'attention weights', determining how much 'focus' the model should place on specific tokens in a sequence relative to others.

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$



### 3.2. Token Prediction (The Output Head)

During the decoding phase, the final linear layer of an LLM produces a vector of logits corresponding to the model's entire vocabulary. Softmax transforms these logits into probabilities, enabling the model to sample the most likely next token (e.g., via greedy search or nucleus sampling).



## 4. Seminal Literature

To consult the primary sources regarding the development and standardisation of Softmax, refer to the following publications:



* \*\*Bridle, J. S. (1989).\*\* \*Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition.\* URL: \[https://link.springer.com/chapter/10.1007/978-3-642-75100-4\_13](https://link.springer.com/chapter/10.1007/978-3-642-75100-4\_13)



* \*\*Luce, R. D. (1959).\*\* \*Individual Choice Behavior: A Theoretical Analysis.\* (Early foundational work on the 'Luce's choice axiom' related to Softmax).

    URL: \[https://psycnet.apa.org/record/1959-08955-000](https://psycnet.apa.org/record/1959-08955-000)

* \*\*Vaswani, et al. (2017).\*\* \*Attention Is All You Need.\* (The standard implementation in LLMs).

    URL: \[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)


