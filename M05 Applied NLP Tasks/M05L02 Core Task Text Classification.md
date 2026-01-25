# Chapter 5.2: Core Task: Text Classification

## 1. Binary vs. Multi-Class Classification
Text classification is the foundation of automated content management. Its objective is to map an input sequence to a set of predefined labels. 
- **Binary Classification** involves exactly two mutually exclusive options (e.g., Spam or Not-Spam). 
- **Multi-Class Classification** involves selecting exactly one category from a list of $N$ options. 
- **Multi-Label Classification** (notably different) allows an input to be assigned to multiple categories simultaneously (e.g., tagging a news article as both "Economy" and "Politics").

## 2. Traditional Methods (Naive Bayes, SVM)
Before the era of deep learning, classification was driven by statistical models operating on TF-IDF word frequencies. 
- **Naive Bayes**: Uses Bayes' Theorem to calculate the probability of a class based on word features, assuming independence between them. While "naive," it is exceptionally fast and remains a robust baseline.
- **Support Vector Machines (SVM)**: A geometric approach that identifies the "Hyperplane" that best separates classes in high-dimensional space. These models served as the industrial standard for decade due to their high accuracy on smaller, structured datasets.

## 3. Deep Learning for Classification (CNN, Transformer)
The current state-of-the-art utilizes non-linear neural networks to capture deep context.
- **CNNs (Convolutional Neural Networks)**: Once popular for their ability to detect "local" word patterns (e.g., detecting specific phrases that indicate toxicity).
- **Transformers (BERT)**: The modern standard. By using the `[CLS]` token and the self-attention mechanism, the model captures global document intent. As detailed by Raschka, fine-tuning a BERT-style encoder on a specialized dataset produces higher precision than any previous methodology.

## 4. Zero-Shot and Few-Shot Classification
A revolutionary capability of Large Language Models (LLMs) is the ability to classify text **Without Specialized Training**. 
- **Zero-Shot**: Providing a prompt like *"Does this sentence contain a technical bug report? Yes/No"* allows the model to use its general pre-trained reasoning to classify inputs it has never seen before.
- **Few-Shot**: providing 3-5 labeled examples in the prompt to define the desired boundary, enabling rapid deployment of new classification taxonomies in minutes rather than months.

## 5. Evaluation Metrics (Precision, Recall, F1-Score)
In production, simple accuracy is often misleading. We rely on the **Confusion Matrix** to derive:
- **Precision**: $\frac{TP}{TP + FP}$ — of all positive predictions, how many were correct? (Crucial for avoiding false alarms).
- **Recall**: $\frac{TP}{TP + FN}$ — how many of the actual positive cases did the model catch? (Crucial for safety and compliance).
- **F1-Score**: The harmonic mean that provides a balanced view of performance on imbalanced datasets.
By mastering these metrics, a developer ensures that the classification engine is not just fluent, but industrially reliable.
