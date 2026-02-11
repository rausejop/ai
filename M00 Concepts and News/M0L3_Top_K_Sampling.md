# Module 0, Lesson 3: Top-K Sampling in Large Language Models

## Learning Objectives

By the end of this lesson, students will be able to:
1. Explain the mathematical formulation and algorithmic implementation of top-k sampling
2. Analyse the historical development from greedy decoding to stochastic sampling methods
3. Evaluate the trade-offs between determinism and diversity in text generation
4. Implement efficient top-k sampling algorithms with various optimisations
5. Compare top-k sampling with alternative decoding strategies

---

## 1. The Problem: Deterministic Decoding Limitations

### 1.1 Greedy Decoding

**Definition:** At each step, select the token with the highest probability:

$$y_t = \arg\max_{i} P(y_i | y_{<t})$$

**Problems:**
1. **Repetition:** Models often get stuck in loops ("very very very very...")
2. **Lack of Diversity:** Same input always produces identical output
3. **Local Optima:** Greedy choices may prevent globally optimal sequences
4. **Unnatural Text:** Human language has inherent variability

**Example:**

Prompt: "The weather today is"

Greedy output: "The weather today is very nice and sunny and warm and pleasant and beautiful and..."

### 1.2 Beam Search

**Definition:** Maintain top-k hypotheses at each step, expanding each:

$$\text{Score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{<i})$$

**Problems:**
1. **Generic Output:** Tends toward safe, common phrases
2. **Computational Cost:** $O(k \times |V|)$ per step, where $|V|$ is vocabulary size
3. **Still Deterministic:** Same beam width → same output
4. **Poor for Open-Ended Generation:** Works for translation, fails for creative tasks

---

## 2. Top-K Sampling: Core Concept

### 2.1 Mathematical Formulation

**Algorithm:**
1. Compute probability distribution: $P(y_i | y_{<t})$ for all tokens $i \in V$
2. Select the $k$ tokens with highest probabilities
3. Renormalise probabilities over these $k$ tokens
4. Sample from the renormalised distribution

**Formal Definition:**

Let $V_k = \{i_1, i_2, \ldots, i_k\}$ be the indices of the top-k tokens sorted by probability.

$$P_{\text{top-k}}(y_i | y_{<t}) = \begin{cases}
\frac{P(y_i | y_{<t})}{\sum_{j \in V_k} P(y_j | y_{<t})} & \text{if } i \in V_k \\
0 & \text{otherwise}
\end{cases}$$

Then sample: $y_t \sim P_{\text{top-k}}(\cdot | y_{<t})$

### 2.2 Pseudocode

```
function TopKSampling(logits, k):
    # Step 1: Apply softmax to get probabilities
    probs = softmax(logits)
    
    # Step 2: Find top-k indices
    top_k_indices = argsort(probs, descending=True)[:k]
    
    # Step 3: Extract top-k probabilities
    top_k_probs = probs[top_k_indices]
    
    # Step 4: Renormalise
    top_k_probs = top_k_probs / sum(top_k_probs)
    
    # Step 5: Sample from renormalised distribution
    sampled_index = categorical_sample(top_k_probs)
    
    return top_k_indices[sampled_index]
```

---

## 3. Historical Development

### 3.1 Early Stochastic Methods (Pre-2018)

**Random Sampling (Temperature-Only):**
- Used in early RNN language models
- Problem: Samples from entire vocabulary, including very low-probability tokens
- Result: Frequent nonsensical outputs

**Reference:**
- Graves, A. (2013). "Generating Sequences With Recurrent Neural Networks." *arXiv preprint arXiv:1308.0850*.

### 3.2 Emergence of Top-K (2018)

Top-k sampling became a "popular alternative sampling procedure" around 2018, as noted in the nucleus sampling paper:

**Reference:**
- Fan, A., Lewis, M., & Dauphin, Y. (2018). "Hierarchical Neural Story Generation." *Proceedings of ACL 2018*. [https://arxiv.org/abs/1805.04833](https://arxiv.org/abs/1805.04833)

This paper used top-k sampling for story generation, demonstrating improved coherence over pure random sampling.

### 3.3 Standardisation in GPT-2 (2019)

**OpenAI's GPT-2** popularised top-k sampling as a standard decoding strategy:

**Reference:**
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Blog*. [https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

GPT-2 used $k = 40$ as default, establishing this as a common baseline.

### 3.4 Critical Analysis (2019)

**Holtzman et al. (2019)** provided the first comprehensive analysis of top-k sampling's limitations:

**Reference:**
- Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019). "The Curious Case of Neural Text Degeneration." *ICLR 2020*. [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)

**Key Findings:**
1. Fixed $k$ is suboptimal: sometimes too restrictive, sometimes too permissive
2. Introduced **nucleus sampling (top-p)** as an adaptive alternative
3. Demonstrated that top-k with $k=640$ approaches full sampling

---

## 4. Mathematical Analysis

### 4.1 Effect on Probability Distribution

**Original Distribution:**

Suppose we have vocabulary size $|V| = 50,000$ with probabilities:
- Top token: $P(y_1) = 0.3$
- Next 9 tokens: $P(y_i) = 0.05$ each ($i = 2, \ldots, 10$)
- Remaining 49,990 tokens: $P(y_i) \approx 0.00004$ each

**After Top-K with $k=10$:**

$$P_{\text{top-10}}(y_1) = \frac{0.3}{0.3 + 9 \times 0.05} = \frac{0.3}{0.75} = 0.4$$

$$P_{\text{top-10}}(y_i) = \frac{0.05}{0.75} = 0.067 \quad (i = 2, \ldots, 10)$$

$$P_{\text{top-10}}(y_i) = 0 \quad (i > 10)$$

**Observation:** Renormalisation increases all top-k probabilities proportionally.

### 4.2 Entropy Analysis

**Shannon Entropy Before Top-K:**

$$H_{\text{full}} = -\sum_{i=1}^{|V|} P(y_i) \log P(y_i)$$

**Shannon Entropy After Top-K:**

$$H_{\text{top-k}} = -\sum_{i=1}^{k} P_{\text{top-k}}(y_i) \log P_{\text{top-k}}(y_i)$$

**Relationship:**

$$H_{\text{top-k}} \leq H_{\text{full}}$$

Top-k sampling **reduces entropy** by eliminating low-probability tokens.

**Extreme Cases:**
- $k = 1$: $H = 0$ (deterministic, equivalent to greedy)
- $k = |V|$: $H = H_{\text{full}}$ (no filtering)

### 4.3 Computational Complexity

**Naive Implementation:**
1. Softmax: $O(|V|)$
2. Full sort: $O(|V| \log |V|)$
3. Renormalisation: $O(k)$
4. Sampling: $O(k)$

**Total:** $O(|V| \log |V|)$ - dominated by sorting

**Optimised Implementation (Partial Sort):**
1. Softmax: $O(|V|)$
2. Top-k selection (quickselect): $O(|V|)$ average case
3. Sort top-k: $O(k \log k)$
4. Renormalisation: $O(k)$

**Total:** $O(|V| + k \log k)$ - much faster for small $k$

---

## 5. Practical Implementation

### 5.1 NumPy Implementation

```python
import numpy as np

def top_k_sampling(logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
    """
    Sample from top-k tokens with temperature scaling.
    
    Args:
        logits: Array of raw scores (shape: [vocab_size])
        k: Number of top candidates to consider
        temperature: Temperature scaling factor
    
    Returns:
        Sampled token index
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Get top-k indices (using argpartition for efficiency)
    if k >= len(logits):
        top_k_indices = np.arange(len(logits))
    else:
        # Partial sort: O(n) instead of O(n log n)
        top_k_indices = np.argpartition(scaled_logits, -k)[-k:]
    
    # Extract top-k logits
    top_k_logits = scaled_logits[top_k_indices]
    
    # Apply softmax to top-k
    top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    
    # Sample from renormalised distribution
    sampled_index = np.random.choice(len(top_k_indices), p=top_k_probs)
    
    return top_k_indices[sampled_index]


# Example usage
vocab_size = 50000
logits = np.random.randn(vocab_size)
logits[100] = 5.0  # Make token 100 highly likely

print("Sampling 1000 times with different k values:")
for k in [1, 10, 50, 100]:
    samples = [top_k_sampling(logits, k) for _ in range(1000)]
    unique_tokens = len(set(samples))
    print(f"k={k:3d}: {unique_tokens:4d} unique tokens sampled")
```

### 5.2 PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def top_k_sampling_torch(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Efficient top-k sampling using PyTorch.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size) or (vocab_size,)
        k: Number of top candidates
        temperature: Temperature scaling
    
    Returns:
        Sampled token indices
    """
    # Handle single sample or batch
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Get top-k values and indices
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
    
    # Apply softmax to top-k
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from categorical distribution
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
    
    # Map back to original vocabulary indices
    sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices)
    
    if squeeze_output:
        return sampled_tokens.squeeze()
    return sampled_tokens


# Batch example
batch_size = 4
vocab_size = 50257  # GPT-2 vocabulary size
logits = torch.randn(batch_size, vocab_size)

samples = top_k_sampling_torch(logits, k=50, temperature=0.8)
print(f"Sampled tokens: {samples}")
```

### 5.3 Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with top-k sampling
outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,          # Enable sampling
    top_k=50,                # Top-k value
    temperature=0.8,         # Temperature scaling
    num_return_sequences=3   # Generate 3 different outputs
)

for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\nSample {i+1}:")
    print(text)
```

---

## 6. Parameter Selection and Tuning

### 6.1 Common Values Across Models

| Model | Default k | Range | Notes |
|-------|-----------|-------|-------|
| **GPT-2** | 40 | [1, vocab_size] | Original OpenAI implementation |
| **GPT-3** | 0 (disabled) | [0, vocab_size] | Uses top-p by default |
| **Gemini Flash 2.0** | 64 | Fixed | Combined with top-p=0.95 |
| **LLaMA** | 50 | [1, vocab_size] | Recommended for chat |
| **Claude** | N/A | N/A | Uses top-p exclusively |

### 6.2 Recommended Values by Use Case

| Use Case | Recommended k | Reasoning |
|----------|---------------|-----------|
| **Code Generation** | 10-20 | Limited valid syntax options |
| **Factual Q&A** | 5-10 | Prefer high-confidence answers |
| **Creative Writing** | 50-100 | Allow diverse vocabulary |
| **Dialogue** | 30-50 | Balance naturalness and coherence |
| **Translation** | 10-30 | Constrained by source text |
| **Summarisation** | 20-40 | Moderate diversity |

### 6.3 Interaction with Temperature

**Combined Effect:**

```python
# Low temperature + low k: Very deterministic
sample = top_k_sampling(logits, k=5, temperature=0.3)

# High temperature + high k: Very diverse
sample = top_k_sampling(logits, k=100, temperature=1.5)

# Balanced: Moderate both
sample = top_k_sampling(logits, k=50, temperature=0.8)
```

**Rule of Thumb:**
- If using low temperature (< 0.5), use smaller k (10-30)
- If using high temperature (> 1.0), use larger k (50-100)
- For temperature ≈ 1.0, k ≈ 40-50 works well

---

## 7. Worked Examples

### Example 1: Manual Top-K Calculation

**Given:** Vocabulary of 10 tokens with probabilities:

| Token | Probability |
|-------|-------------|
| "the" | 0.25 |
| "a" | 0.20 |
| "is" | 0.15 |
| "of" | 0.10 |
| "and" | 0.08 |
| "to" | 0.07 |
| "in" | 0.06 |
| "that" | 0.04 |
| "it" | 0.03 |
| "for" | 0.02 |

**Task:** Apply top-k sampling with $k=5$

**Solution:**

**Step 1:** Select top-5 tokens
- "the" (0.25), "a" (0.20), "is" (0.15), "of" (0.10), "and" (0.08)

**Step 2:** Sum of top-5 probabilities
- $0.25 + 0.20 + 0.15 + 0.10 + 0.08 = 0.78$

**Step 3:** Renormalise
- "the": $0.25 / 0.78 = 0.321$
- "a": $0.20 / 0.78 = 0.256$
- "is": $0.15 / 0.78 = 0.192$
- "of": $0.10 / 0.78 = 0.128$
- "and": $0.08 / 0.78 = 0.103$

**Step 4:** Sample from renormalised distribution

**Verification:** $0.321 + 0.256 + 0.192 + 0.128 + 0.103 = 1.000$ ✓

### Example 2: Comparing Different k Values

**Scenario:** Next word prediction after "The cat"

**Original Probabilities (top 10):**
1. "sat" - 0.30
2. "is" - 0.15
3. "was" - 0.12
4. "jumped" - 0.08
5. "ran" - 0.06
6. "meowed" - 0.05
7. "slept" - 0.04
8. "ate" - 0.03
9. "purred" - 0.02
10. "hissed" - 0.01

**Case A: k=1 (Greedy)**
- Always select "sat"
- Output: "The cat sat sat sat..." (repetitive)

**Case B: k=3**
- Candidates: "sat" (0.30), "is" (0.15), "was" (0.12)
- Renormalised: "sat" (0.526), "is" (0.263), "was" (0.211)
- Output: Mostly "sat", sometimes "is" or "was"

**Case C: k=10**
- All 10 tokens available
- Renormalised: Similar to original (sum ≈ 0.86)
- Output: Diverse, including rare words like "hissed"

### Example 3: Entropy Calculation

**Given:** Top-5 probabilities from Example 1

**Original Entropy (top-5 only):**

$$H_{\text{orig}} = -(0.25 \log 0.25 + 0.20 \log 0.20 + 0.15 \log 0.15 + 0.10 \log 0.10 + 0.08 \log 0.08)$$

$$H_{\text{orig}} = -(0.25 \times (-1.386) + 0.20 \times (-1.609) + 0.15 \times (-1.897) + 0.10 \times (-2.303) + 0.08 \times (-2.526))$$

$$H_{\text{orig}} = 1.485 \text{ nats}$$

**After Renormalisation:**

$$H_{\text{top-5}} = -(0.321 \log 0.321 + 0.256 \log 0.256 + 0.192 \log 0.192 + 0.128 \log 0.128 + 0.103 \log 0.103)$$

$$H_{\text{top-5}} = 1.547 \text{ nats}$$

**Observation:** Entropy slightly increases due to renormalisation flattening the distribution.

---

## 8. Advanced Topics

### 8.1 Adaptive Top-K

Dynamically adjust $k$ based on distribution shape:

```python
def adaptive_top_k(logits: np.ndarray, 
                   min_k: int = 10, 
                   max_k: int = 100,
                   entropy_threshold: float = 3.0) -> int:
    """
    Adjust k based on entropy of probability distribution.
    High entropy (uncertain) → larger k
    Low entropy (confident) → smaller k
    """
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Linear interpolation based on entropy
    k = int(min_k + (max_k - min_k) * min(entropy / entropy_threshold, 1.0))
    
    return k
```

### 8.2 Top-K with Threshold

Combine top-k with minimum probability threshold:

```python
def top_k_with_threshold(logits: np.ndarray, 
                         k: int, 
                         min_prob: float = 0.01) -> int:
    """
    Apply top-k, but exclude tokens below minimum probability.
    """
    probs = softmax(logits)
    
    # Get top-k
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices]
    
    # Filter by threshold
    valid_mask = top_k_probs >= min_prob
    filtered_indices = top_k_indices[valid_mask]
    filtered_probs = top_k_probs[valid_mask]
    
    # Renormalise and sample
    filtered_probs = filtered_probs / np.sum(filtered_probs)
    sampled_idx = np.random.choice(len(filtered_indices), p=filtered_probs)
    
    return filtered_indices[sampled_idx]
```

### 8.3 Typical Sampling (Top-K Variant)

Select tokens within a certain probability mass of the mode:

```python
def typical_sampling(logits: np.ndarray, tau: float = 0.9) -> int:
    """
    Sample from tokens with 'typical' information content.
    
    Reference: Meister et al. (2022) "Typical Decoding for Natural Language Generation"
    """
    probs = softmax(logits)
    
    # Compute information content
    info = -np.log(probs)
    expected_info = np.sum(probs * info)
    
    # Select tokens close to expected information
    deviation = np.abs(info - expected_info)
    typical_indices = np.argsort(deviation)
    
    # Take top tokens until cumulative probability exceeds tau
    cumsum = 0
    k = 0
    while cumsum < tau and k < len(typical_indices):
        cumsum += probs[typical_indices[k]]
        k += 1
    
    # Sample from typical set
    typical_probs = probs[typical_indices[:k]]
    typical_probs = typical_probs / np.sum(typical_probs)
    
    sampled_idx = np.random.choice(k, p=typical_probs)
    return typical_indices[sampled_idx]
```

---

## 9. Limitations and Criticisms

### 9.1 Fixed k Problem

**Issue:** Optimal $k$ varies by context

**Example:**

Prompt 1: "The capital of France is"
- Model is very confident: "Paris" has 0.95 probability
- Optimal $k$: 1-3 (high confidence context)

Prompt 2: "The weather tomorrow will be"
- Model is uncertain: top token has 0.15 probability
- Optimal $k$: 50-100 (high uncertainty context)

**Solution:** Use nucleus sampling (top-p) instead, which adapts automatically

### 9.2 Truncation Artifacts

**Issue:** Hard cutoff can exclude plausible tokens

**Example:**

Probabilities: [0.11, 0.10, 0.09, 0.08, 0.07, ...]

With $k=3$: Tokens 4 and 5 (0.08, 0.07) are excluded despite being nearly as likely as token 3 (0.09)

### 9.3 Computational Overhead

**Issue:** Sorting is expensive for large vocabularies

**Mitigation:**
- Use partial sorting algorithms (quickselect)
- Implement on GPU with parallel sorting
- Cache top-k indices when possible

---

## 10. Exercises and Practice Problems

### Exercise 1: Implementation

**Task:** Implement top-k sampling with the following optimisations:
1. Use `np.argpartition` for $O(n)$ selection
2. Add temperature scaling
3. Handle edge cases (k=0, k > vocab_size)
4. Add input validation

### Exercise 2: Empirical Analysis

**Task:** Using GPT-2:
1. Generate 100 samples for the same prompt with $k \in \{1, 10, 50, 100, 500\}$
2. Measure:
   - Unique n-gram count (diversity)
   - Average perplexity (coherence)
   - Repetition rate
3. Plot metrics vs. $k$

### Exercise 3: Mathematical Proof

**Task:** Prove that for any probability distribution $P$ and $k_1 < k_2$:

$$H(\text{top-}k_1) \leq H(\text{top-}k_2)$$

Where $H$ is Shannon entropy.

### Exercise 4: Comparative Study

**Task:** Compare top-k sampling ($k=50$) with:
- Greedy decoding
- Pure random sampling
- Beam search (beam=5)

For the task of story generation. Evaluate on:
- Coherence (human evaluation)
- Diversity (self-BLEU)
- Fluency (perplexity)

---

## 11. Summary and Key Takeaways

1. **Core Concept:** Top-k sampling restricts sampling to the $k$ most probable tokens, balancing diversity and coherence

2. **Historical Evolution:**
   - Pre-2018: Random sampling (too diverse)
   - 2018: Top-k emerges in story generation
   - 2019: GPT-2 popularises $k=40$ as default
   - 2019: Nucleus sampling (top-p) proposed as adaptive alternative

3. **Mathematical Properties:**
   - Reduces entropy compared to full distribution
   - Computational complexity: $O(n + k \log k)$ with optimisation
   - Renormalisation increases all top-k probabilities

4. **Practical Guidelines:**
   - Code generation: $k=10-20$
   - Creative writing: $k=50-100$
   - General chat: $k=30-50$
   - Combine with temperature for fine control

5. **Limitations:**
   - Fixed $k$ doesn't adapt to context
   - Hard cutoff can exclude plausible tokens
   - Superseded by top-p in many applications

6. **Industry Standards:**
   - GPT-2: $k=40$ (historical)
   - Gemini Flash 2.0: $k=64$ (fixed, combined with top-p)
   - Modern practice: Often use top-p instead or in combination

---

## 12. Further Reading

### Foundational Papers
1. Fan, A., Lewis, M., & Dauphin, Y. (2018). "Hierarchical Neural Story Generation"
2. Holtzman, A., et al. (2019). "The Curious Case of Neural Text Degeneration"
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"

### Advanced Topics
1. Meister, C., et al. (2022). "Typical Decoding for Natural Language Generation." *arXiv:2202.00666*
2. Hewitt, J., et al. (2022). "Truncation Sampling as Language Model Desmoothing." *arXiv:2210.15191*

### Comparative Studies
1. Ippolito, D., et al. (2019). "Comparison of Diverse Decoding Methods from Conditional Language Models." *ACL 2019*
2. Zhang, H., et al. (2021). "Trading Off Diversity and Quality in Natural Language Generation." *ACL 2021*

---

*End of Lesson 3*
