# Chapter 8.2: In-Context Learning: Zero-Shot and Few-Shot

## 1. Zero-Shot Learning Explained
In **Zero-Shot Learning**, a model is presented with a task it has never specifically been trained to perform, without any examples provided in the prompt. The model must rely entirely on the high-level semantic reasoning it inherited during its massive pre-training phase. While powerful for general questions, zero-shot prompting can lead to inconsistent formatting or "off-topic" generation in complex industrial tasks, necessitating more structured guidance.

## 2. Few-Shot Learning: The Role of Examples
**Few-Shot Learning** is the process where a developer provide between 3 and 10 examples of **(Input, Output)** pairs before the final user query. These examples act as a "Blueprint," teaching the model the desired tone, length, and technical format. Mathematically, these examples shift the model's attention weights toward the specific sub-domain of the task, significantly increasing the probability of a correct and reliably formatted response.

## 3. Best Practices for Selecting In-Context Examples
The quality of a few-shot prompt is determined by its **Diversity** and **Order**.
- **Diversity**: Examples should cover "Edge Cases" and varying styles to prevent the model from becoming biased toward a single sub-topic.
- **Order Consistency**: Models often suffer from **Recency Bias**, giving more weight to the last example in the list.
- **Label Distribution**: In classification tasks, it is critical to provide an equal number of positive and negative examples to prevent the model from developing a probabilistic bias toward a specific label.

## 4. Demo: Few-Shot Classification
Few-shot prompting allows for the immediate creation of custom classifiers. By providing three examples of "High Urgency" customer emails and three "Low Urgency" ones, a model can accurately classify a seventh, unseen email with higher precision than a zero-shot model. This "Soft Programming" approach allows developers to build functional classifiers in seconds without writing a single line of training code.

## 5. Limitations of In-Context Learning
Despite its power, ICL has three primary technical constraints:
- **Token Cost**: Every example consumes space in the context window, increasing the cost per query.
- **Session Volatility**: The "Learning" is transient; the model "forgets" the examples as soon as the session ends or the context window is cleared.
- **Quadratic Latency**: As the number of examples increases, the computational cost of the attention mechanism ($O(N^2)$) grows, potentially slowing down the response time for real-world users.
By balancing these factors, architects ensure that In-Context Learning serves as a bridge to expert-level AI performance.
