import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute 'Scaled Dot-Product Attention' as used in Transformer LLMs.
    
    Args:
        query: Matrix of Query vectors (n_queries, d_k)
        key: Matrix of Key vectors (n_keys, d_k)
        value: Matrix of Value vectors (n_keys, d_v)
        mask: Optional mask to prevent attention to certain tokens (e.g., padding)
    """
    # 1. Calculate the raw dot product (similar to similarity scores)
    # This results in 'logits' for the attention mechanism
    matmul_qk = np.matmul(query, key.T)

    # 2. THE SCALING FACTOR (The 'Understanding' part)
    # We divide by the square root of the dimension of the keys (d_k).
    # WHY? If d_k is large (e.g., 512), the dot product values grow large.
    # Large inputs to Softmax result in extremely small gradients.
    dk = query.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # 3. Optional Masking
    # Usually used in the decoder to prevent looking at future tokens
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 4. Softmax Application
    # This normalises the scaled scores so they sum to 1.
    # Each row now represents 'how much attention' to pay to each word.
    attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)

    # 5. Final Output
    # Multiply weights by the 'Values' to get the weighted context vector.
    output = np.matmul(attention_weights, value)

    return output, attention_weights

# --- EXAMPLE CASE ---
d_k = 64  # Dimension of the embeddings
# Imagine 3 words (tokens) in a sentence
q = np.random.randn(3, d_k)
k = np.random.randn(3, d_k)
v = np.random.randn(3, d_k)

output, weights = scaled_dot_product_attention(q, k, v)

print(f"Attention Weights (Summing to 1):\n{weights}")
print(f"Scaling factor used: 1/sqrt({d_k}) = {1/np.sqrt(d_k):.4f}")