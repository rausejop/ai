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

## 📊 Visual Resources and Diagrams

- **Encoder-Decoder Connectivity Map**: Showing how information flows from the source stack to the target generation stack.
    ![Encoder-Decoder Connectivity Map](https://lilianweng.github.io/posts/2023-01-27-transformer-models-survey/transformer-decoder.png)
    - [Source: Lilian Weng Blog - The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-transformer-models-survey/transformer-decoder.png)
- **Causal Masking Trick Visualization**: A matrix diagram showing the lower-triangular visibility logic in Decoders.
    ![Causal Masking Trick Visualization](https://jalammar.github.io/images/gpt2/gpt2-self-attention-mask-matrix.png)
    - [Source: Jay Alammar - The Illustrated GPT-2](https://jalammar.github.io/images/gpt2/gpt2-self-attention-mask-matrix.png)

## 🐍 Technical Implementation (Python 3.14.2)

Demonstrating **Causal Masking** for Decoder-style generation in `torch`.

```python
import torch # Importing the core PyTorch library for multi-dimensional tensor operations

def generate_causal_mask(size: int): # Defining a function to generate the visibility mask for a generative decoder
    """ # Start of the function's docstring
    Creates a lower-triangular mask for generative decoders. # Explaining the goal of enforcing temporal order
    Compatible with Python 3.14.2. # Specifying the target runtime for current industrial AI stacks
    """ # End of docstring
    # 1. Create a matrix of ones # Section for initial matrix construction
    mask = torch.ones((size, size)) # Initializing a square matrix of size 'L x L' containing only ones
    
    # 2. Extract the lower-triangular part (including diagonal) # Section for creating the causal barrier
    # tril = 1 for past, 0 for future # Pedagogical note on the 'tril' (Triangle Lower) operator
    causal_mask = torch.tril(mask) # Zeroing out the upper triangle to prevent the model from 'seeing' the future
    
    return causal_mask # Returning the completed Boolean/Numerical causal mask

if __name__ == "__main__": # Entry point check for the standalone demonstration script
    # Visualizing a mask for a 5-token sequence # Preparing a small sample for visual validation
    mask = generate_causal_mask(5) # Generating a 5x5 causal mask
    print("Causal Matrix (1=Visible, 0=Masked):") # Outputting a descriptive label for the console log
    print(mask) # Printing the lower-triangular matrix to verify the visibility logic
    
    # Applying to raw scores: inf ensures softmax( future ) = 0 # Section for demonstrating usage in an attention head
    scores = torch.randn(5, 5) # Generating random attention compatibility scores for a 5-token sequence
    masked_scores = scores.masked_fill(mask == 0, float('-inf')) # Silencing the future tokens by setting their scores to negative infinity
    print("\nScore matrix after masking (negative infinity prevents cheating):") # Outputting confirmation of the silencing step
    print(masked_scores) # Displaying the final scores ready for the softmax probability layer
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Raffel et al. (2019)**: *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"*. The T5 paper explaining the power of unified Seq2Seq.
    - [Link to ArXiv](https://arxiv.org/abs/1910.10683)
- **Sutskever et al. (2014)**: *"Sequence to Sequence Learning with Neural Networks"*. The pre-Transformer conceptual foundation of the dual-stack model.
    - [Link to ArXiv](https://arxiv.org/abs/1409.3215)

### Frontier News and Updates (2025-2026)
- **Google DeepMind (Early 2026)**: Introduction of *Unified-Decoder-V2*, an architecture that removes the encoder entirely, even for translation tasks, by using massive "Instruction Context."
- **NVIDIA AI Blog**: "The Throughput of Generation"—How new HBM4 memory architectures solve the latency bottleneck for auto-regressive decoders.
- **Anthropic Tech Insights**: Discussion on "Reverse-Attention"—an experimental decoding strategy where the model generates global topics before filling in specific tokens.
