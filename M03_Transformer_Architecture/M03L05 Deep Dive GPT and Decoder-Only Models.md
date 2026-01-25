# Chapter 3.5: Deep Dive: GPT and Decoder-Only Models

## The Generative Paradigm and Scaling Logic

The **Generative Pre-trained Transformer (GPT)** series, developed by OpenAI, represents the technical culmination of the decoder-only philosophy. While encoders focus on feature extraction, decoders are optimized for the stochastic process of sequential text generation. The power of GPT derives from two distinct technical sources: the **Causal Attention mechanism** and the **Predictable Dynamics of Scaling**.

### Causal Attention and Auto-regressive Generation

The defining technical constraint of a decoder is its inability to "see" the future during training. This is achieved through **Causal (Masked) Self-Attention**. During the training phase, the model is presented with a full sentence, but for setiap token $i$, the model's attention scores are multiplied by a mask that sets all future tokens ($j > i$) to negative infinity. 

At inference time, the model enters an **Auto-regressive Loop**:
1.  **Input**: A prompt $P$ (sequence of tokens).
2.  **Forward Pass**: The model computes the probability distribution for the *single* next token $x_{t+1}$ over the entire vocabulary.
3.  **Sampling**: A token is chosen (using techniques like Top-K or Nucleus Sampling).
4.  **Feedback**: $x_{t+1}$ is appended to $P$, and the process repeats until a stop token is reached.

### The Scaling Laws of Neural Intelligence

GPT-3 revolutionized the field not through a new architecture, but through unprecedented scale (175 billion parameters). Technical research, specifically the **LLM Scaling Laws** (Kaplan et al., 2020), proved that model performance (loss) follows a strictly predictable power-law relationship with respect to:
- **$N$**: The number of model parameters.
- **$D$**: The size of the dataset.
- **$C$**: The amount of compute (FLOPs) available for training.

This discovery implied that "intelligence" could be treated as an engineering problem—if you increase these three factors in proportion, the model's capability will improve in a straight line on a log-log scale.

### Emergence: In-Context Learning

As GPT models reached the billion-parameter threshold, they began to exhibit **Emergent Abilities** that were not explicitly programmed. Chief among these is **In-Context Learning (ICL)**. An LLM of sufficient scale can perform a task (e.g., translating a specific slang term) simply by being shown two or three examples in the prompt. This "Few-Shot" ability allows the model to behave as a versatile, re-programmable logic engine without ever requiring a single weight update for new tasks.

## Architectural Optimization: KV Caching

To make this auto-regressive generation computationally viable, modern decoders utilize **KV (Key-Value) Caching**. Since tokens at position $1 \dots t$ do not change when predicting token $t+1$, the model stores their Key and Value vectors in memory. This eliminates the need to re-calculate the entire attention history at every step, reducing the inference cost from $O(T^2)$ to $O(T)$, which is essential for providing the low-latency responses required by modern AI applications.
