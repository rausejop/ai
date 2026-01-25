# Chapter 3.3: Encoder vs. Decoder Architectures

## 1. Sequence-to-Sequence (Seq2Seq) Overview
The original Transformer was designed for **Sequence-to-Sequence (Seq2Seq)** tasks, such as machine translation. This architecture consists of two primary components: an **Encoder** that compresses the source text into a dense representation, and a **Decoder** that generates the target text one token at a time. While the field has evolved, this dual-stack logic remains the foundation for almost all generative and understanding tasks in modern AI.

## 2. The Encoder Stack: Learning Representations
The **Encoder** stack (e.g., BERT) is architected for **Natural Language Understanding (NLU)**. Its defining technical characteristic is **Full Bidirectionality**. In an encoder, every token can "look" both forward and backward at its neighbors. This allows the model to develop a deep, holistic representation of the context, making it the ideal choice for tasks like classification, named entity recognition, and sentiment detection.

## 3. The Decoder Stack: Generating Output
The **Decoder** stack (e.g., the GPT lineage) is optimized for **Generative** tasks. Its objective is to predict the "next token" in a sequence. To ensure it doesn't "cheat" during training, the decoder utilizes **Causal (Masked) Self-Attention**, preventing the model from seeing future tokens. 
- **Auto-regression**: Decoders generate text step-by-step; the token generated at time $t$ is appended back into the input to help predict the token at time $t+1$.

## 4. Masked Multi-Head Attention in the Decoder
Inside the decoder layer, a specialized **Look-ahead Mask** is applied to the self-attention scores. Mathematically, this mask sets the attention values of all "future" positions to negative infinity before the softmax operation. This physical barrier ensures that the model's prediction at position $i$ depends only on the tokens at positions $1 \dots i$. Furthermore, in Seq2Seq models, the decoder features a unique **Encoder-Decoder Cross-Attention** layer, which allows the generative process to "look back" at the encoder's original summary, ensuring that the generated output remains faithful to the source intent.
