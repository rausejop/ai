# Chapter 5.3: Understanding User Intent: Sentiment Analysis

## 1. Aspect-Based vs. Document-Level Sentiment
Sentiment analysis is a specialized form of classification that identifies subjective intent. 
- **Document-Level**: Capturing the aggregate "vibe" of a long review.
- **Aspect-Based Sentiment Analysis (ABSA)**: The high-resolution industry standard. It identifies sentiment relative to specific entities. In the sentence *"The phone's camera is amazing, but the battery life is terrible,"* ABSA correctly attributes POSITIVE sentiment to "Camera" and NEGATIVE to "Battery." This granularity is essential for actionable brand intelligence.

## 2. Lexicon-Based Approaches
Early systems used **Lexicons**—dictionaries where each word is assigned a polarity score (e.g., "happy" = +1.2). While fast and highly explainable (you can see exactly which words triggered the score), these systems are brittle. They fail to understand context-dependent polarity and often mistake descriptive words for emotional ones.

## 3. Sentiment using Pre-trained Embeddings
The transition to **Vector Embeddings** (Module 02) allowed sentiment models to analyze meaning rather than just keywords. By projecting a sentence into latent space, a model can detect that "Stellar" and "Extraordinary" share the same sentiment vector as "Good," even if it has never seen those specific words in a sentiment training set. Modern Transformer-based sentiment heads use these embeddings to achieve high accuracy on diverse, informal social media text.

## 4. Handling Negation and Sarcasm
The primary technical challenge in sentiment is the detection of **Non-Literal Intent**. 
- **Negation**: Understanding that "not good" is negative. While simple for humans, older models often fixated on the word "good."
- **Sarcasm**: Identifying the intent in *"I just love it when my computer crashes!"* This requires the model to identify the semantic mismatch between the positive word "love" and the negative event "crashes." Transformers resolve this by using document-wide context windows to identify irony and nuanced subtext.

## 5. Applications in Market Research
In the modern economy, sentiment analysis serves as the primary sensor for **Consumer Trend Tracking**. By processing millions of reviews and social media posts, organizations can correlate public sentiment with stock price movements, brand reputation, and product success. In enterprise support systems, high-urgency/negative-sentiment tickets are automatically escalated, transforming linguistic probability into immediate operational efficiency.
