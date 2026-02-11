# Module 0, Lesson 2: Temperature Scaling in Large Language Models

## Learning Objectives

By the end of this lesson, students will be able to:
1. Derive the mathematical relationship between temperature and probability distributions
2. Analyse the impact of temperature on model output diversity and determinism
3. Apply temperature scaling to control generation behaviour in different use cases
4. Evaluate the trade-offs between creativity and coherence at different temperature values
5. Implement temperature-scaled sampling algorithms with practical examples

---

## 1. Theoretical Foundations

### 1.1 Origins in Statistical Mechanics

The concept of temperature in neural networks is directly borrowed from **thermodynamics** and **statistical mechanics**. In Boltzmann's formulation, temperature ($T$) controls the probability distribution of particles across energy states.

**Boltzmann Distribution with Temperature:**

$$P(i) = \frac{e^{-E_i / kT}}{\sum_{j} e^{-E_j / kT}}$$

**Physical Interpretation:**
- **Low Temperature ($T \to 0$):** System freezes into the lowest energy state (deterministic)
- **High Temperature ($T \to \infty$):** All states become equally probable (maximum entropy)
- **Intermediate Temperature:** Balanced exploration of states

### 1.2 Adaptation to Neural Networks

In the context of neural networks, we modify the Softmax function to include a temperature parameter:

$$P(y_i | \mathbf{x}) = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}}$$

Where:
- $z_i$ = Logit (raw score) for class/token $i$
- $T$ = Temperature parameter ($T > 0$)
- $P(y_i | \mathbf{x})$ = Probability of output $i$ given input $\mathbf{x}$

**Key Insight:** Temperature acts as a scaling factor applied to logits **before** the Softmax operation.

---

## 2. Mathematical Analysis

### 2.1 Effect on Probability Distribution

Consider logits $\mathbf{z} = [z_1, z_2, z_3]$. Let's analyse how temperature affects the resulting probabilities.

**Case 1: $T = 1$ (Standard Softmax)**

$$P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

This is the baseline case with no temperature scaling.

**Case 2: $T < 1$ (Low Temperature - Sharpening)**

$$P(y_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

Since $T < 1$, dividing by $T$ **increases** the magnitude of logits:
- If $z_i = 2$ and $T = 0.5$, then $z_i / T = 4$
- Larger logits → larger exponentials → more peaked distribution

**Case 3: $T > 1$ (High Temperature - Flattening)**

$$P(y_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

Since $T > 1$, dividing by $T$ **decreases** the magnitude of logits:
- If $z_i = 2$ and $T = 2$, then $z_i / T = 1$
- Smaller logits → smaller exponentials → flatter distribution

### 2.2 Limiting Behaviour

**As $T \to 0$:**

$$\lim_{T \to 0} P(y_i) = \begin{cases}
1 & \text{if } z_i = \max(\mathbf{z}) \\
0 & \text{otherwise}
\end{cases}$$

This becomes **argmax** (greedy decoding) - completely deterministic.

**As $T \to \infty$:**

$$\lim_{T \to \infty} P(y_i) = \frac{1}{n}$$

All tokens become equally probable (uniform distribution) - maximum randomness.

### 2.3 Entropy Analysis

The **Shannon entropy** of a probability distribution measures its uncertainty:

$$H(P) = -\sum_{i=1}^{n} P(y_i) \log P(y_i)$$

**Relationship with Temperature:**
- **Low $T$:** Low entropy (peaked distribution, high certainty)
- **High $T$:** High entropy (flat distribution, high uncertainty)

**Mathematical Proof:**

For a distribution with logits $\mathbf{z}$, the entropy as a function of temperature is:

$$H(T) = \log\left(\sum_{j} e^{z_j / T}\right) - \frac{1}{T} \sum_{i} P_T(y_i) z_i$$

Where $P_T(y_i)$ denotes the temperature-scaled probability.

**Derivative:**

$$\frac{dH}{dT} = \frac{1}{T^2} \sum_{i} P_T(y_i) (z_i - \mathbb{E}[z])^2 \geq 0$$

Since this is always non-negative, **entropy increases monotonically with temperature**.

---

## 3. Practical Applications in LLMs

### 3.1 Gemini Flash 2.0 Default Parameters

According to Google's official documentation:

| Parameter | Default Value | Range | Description |
|-----------|---------------|-------|-------------|
| **Temperature** | 1.0 | [0.0, 2.0] | Controls randomness in sampling |
| **Top-P** | 0.95 | [0.0, 1.0] | Nucleus sampling threshold |
| **Top-K** | 64 | Fixed | Number of top candidates considered |

**Source:** Google AI Vertex AI Documentation (2024)

### 3.2 Use Case Recommendations

| Temperature Range | Classification | Behaviour | Recommended Use Cases |
|-------------------|----------------|-----------|----------------------|
| **0.0 - 0.3** | Deterministic | Highly focused, repetitive, logical | Code generation, mathematical reasoning, factual Q&A, data extraction |
| **0.4 - 0.7** | Conservative | Coherent, predictable, safe | Technical documentation, translations, summaries, educational content |
| **0.8 - 1.0** | Balanced | Natural, varied, conversational | General chatbots, dialogue systems, customer support |
| **1.1 - 1.5** | Creative | Diverse, unexpected, exploratory | Creative writing, brainstorming, poetry, storytelling |
| **1.6 - 2.0** | Experimental | Highly random, potentially incoherent | Artistic experiments, idea generation (with post-filtering) |

**Warning:** Temperatures above 2.0 typically produce nonsensical output due to excessive randomness.

### 3.3 Industry Best Practices

**OpenAI GPT Models:**
- Default: $T = 1.0$
- Range: $[0.0, 2.0]$
- Recommendation: $T = 0.7$ for most applications

**Anthropic Claude:**
- Default: $T = 1.0$
- Range: $[0.0, 1.0]$
- Recommendation: $T = 0.5$ for factual tasks

**Google Gemini:**
- Default: $T = 1.0$
- Range: $[0.0, 2.0]$
- Recommendation: $T = 0.2$ for code, $T = 0.9$ for creative tasks

---

## 4. Historical Development

### 4.1 Early Neural Networks (1990s)

Temperature scaling was first used in **Boltzmann Machines** and **Simulated Annealing** algorithms:

**Seminal Reference:**
- Hinton, G. E., & Sejnowski, T. J. (1986). "Learning and relearning in Boltzmann machines." In *Parallel Distributed Processing* (Vol. 1, pp. 282-317). MIT Press.

### 4.2 Knowledge Distillation (2015)

**Geoffrey Hinton** popularised temperature scaling for **knowledge distillation**, where a smaller "student" model learns from a larger "teacher" model.

**Temperature-Scaled Distillation Loss:**

$$\mathcal{L}_{\text{distill}} = \text{KL}\left(P_{\text{teacher}}(T) \parallel P_{\text{student}}(T)\right)$$

Where both distributions use the same temperature $T > 1$ to soften the probabilities.

**Seminal Reference:**
- Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *arXiv preprint arXiv:1503.02531*. [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

### 4.3 Modern Language Models (2017-Present)

The **Transformer architecture** (Vaswani et al., 2017) standardised the use of temperature in text generation:

**Seminal Reference:**
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**Evolution Timeline:**
- **2017:** Transformers introduce scaled attention (related concept)
- **2018:** GPT-1 uses temperature for controlled generation
- **2019:** GPT-2 popularises temperature + top-k/top-p sampling
- **2020:** GPT-3 establishes temperature as standard hyperparameter
- **2023-2024:** Gemini, Claude, and other models refine default values

---

## 5. Mathematical Derivations and Proofs

### 5.1 Gradient of Temperature-Scaled Softmax

For backpropagation through temperature-scaled Softmax:

$$\frac{\partial P_T(y_i)}{\partial z_j} = \frac{1}{T} P_T(y_i) (\delta_{ij} - P_T(y_j))$$

Where $\delta_{ij}$ is the Kronecker delta.

**Proof:**

Let $P_i = P_T(y_i) = \frac{e^{z_i/T}}{\sum_k e^{z_k/T}}$

For $i = j$:
$$\frac{\partial P_i}{\partial z_i} = \frac{1}{T} \left( \frac{e^{z_i/T} \sum_k e^{z_k/T} - e^{z_i/T} \cdot e^{z_i/T}}{(\sum_k e^{z_k/T})^2} \right) = \frac{1}{T} P_i (1 - P_i)$$

For $i \neq j$:
$$\frac{\partial P_i}{\partial z_j} = \frac{1}{T} \left( \frac{0 - e^{z_i/T} \cdot e^{z_j/T}}{(\sum_k e^{z_k/T})^2} \right) = -\frac{1}{T} P_i P_j$$

### 5.2 Relationship to Cross-Entropy Loss

The cross-entropy loss with temperature scaling:

$$\mathcal{L} = -\log P_T(y_{\text{true}}) = -\frac{z_{\text{true}}}{T} + \log\left(\sum_j e^{z_j/T}\right)$$

**Gradient with respect to logits:**

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{T}(P_T(y_i) - \mathbb{1}[i = y_{\text{true}}])$$

**Observation:** Temperature scales the gradient magnitude. Lower temperature → larger gradients.

---

## 6. Worked Examples

### Example 1: Temperature Effect on Simple Distribution

**Given:** Logits $\mathbf{z} = [1.0, 2.0, 3.0]$

**Task:** Compute probabilities for $T \in \{0.5, 1.0, 2.0\}$

**Solution:**

**Case A: $T = 0.5$ (Low Temperature)**

Scaled logits: $[2.0, 4.0, 6.0]$

$$P(y_1) = \frac{e^{2.0}}{e^{2.0} + e^{4.0} + e^{6.0}} = \frac{7.39}{7.39 + 54.60 + 403.43} = 0.016$$

$$P(y_2) = \frac{54.60}{465.42} = 0.117$$

$$P(y_3) = \frac{403.43}{465.42} = 0.867$$

**Result:** $[0.016, 0.117, 0.867]$ - Highly peaked!

**Case B: $T = 1.0$ (Standard)**

Scaled logits: $[1.0, 2.0, 3.0]$ (unchanged)

$$P(y_1) = \frac{e^{1.0}}{e^{1.0} + e^{2.0} + e^{3.0}} = \frac{2.72}{30.19} = 0.090$$

$$P(y_2) = \frac{7.39}{30.19} = 0.245$$

$$P(y_3) = \frac{20.09}{30.19} = 0.665$$

**Result:** $[0.090, 0.245, 0.665]$ - Moderate distribution

**Case C: $T = 2.0$ (High Temperature)**

Scaled logits: $[0.5, 1.0, 1.5]$

$$P(y_1) = \frac{e^{0.5}}{e^{0.5} + e^{1.0} + e^{1.5}} = \frac{1.65}{1.65 + 2.72 + 4.48} = 0.186$$

$$P(y_2) = \frac{2.72}{8.85} = 0.307$$

$$P(y_3) = \frac{4.48}{8.85} = 0.506$$

**Result:** $[0.186, 0.307, 0.506]$ - Flatter distribution

**Visualisation:**

| Token | $T=0.5$ | $T=1.0$ | $T=2.0$ |
|-------|---------|---------|---------|
| 1 | 1.6% | 9.0% | 18.6% |
| 2 | 11.7% | 24.5% | 30.7% |
| 3 | 86.7% | 66.5% | 50.6% |

**Observation:** As temperature increases, the distribution becomes more uniform.

### Example 2: Real-World Text Generation

**Scenario:** Generating the next word after "The cat sat on the"

**Model Logits (top 5):**
- "mat": 3.2
- "floor": 2.8
- "chair": 2.5
- "table": 2.3
- "roof": 1.9

**Temperature $T = 0.2$ (Deterministic):**
- "mat": 0.89
- "floor": 0.07
- "chair": 0.03
- "table": 0.01
- "roof": 0.00

**Output:** Almost always "mat" (boring but safe)

**Temperature $T = 1.0$ (Balanced):**
- "mat": 0.38
- "floor": 0.25
- "chair": 0.18
- "table": 0.15
- "roof": 0.04

**Output:** Varied but coherent

**Temperature $T = 1.5$ (Creative):**
- "mat": 0.28
- "floor": 0.23
- "chair": 0.20
- "table": 0.18
- "roof": 0.11

**Output:** More diverse, including less common options like "roof"

---

## 7. Implementation Guide

### 7.1 Python Implementation

```python
import numpy as np

def temperature_scaled_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling to logits before Softmax.
    
    Args:
        logits: Array of raw scores
        temperature: Scaling factor (T > 0)
    
    Returns:
        Temperature-scaled probability distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply numerically stable Softmax
    shifted = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(shifted)
    
    return exp_logits / np.sum(exp_logits)


def sample_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a token index using temperature-scaled probabilities.
    
    Args:
        logits: Array of raw scores
        temperature: Scaling factor
    
    Returns:
        Sampled token index
    """
    probs = temperature_scaled_softmax(logits, temperature)
    return np.random.choice(len(logits), p=probs)


# Example usage
logits = np.array([1.0, 2.0, 3.0, 2.5, 1.5])

print("Temperature Comparison:")
for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
    probs = temperature_scaled_softmax(logits, temp)
    print(f"T={temp:.1f}: {probs}")
    
# Sampling demonstration
print("\nSampling 1000 times with different temperatures:")
for temp in [0.5, 1.0, 1.5]:
    samples = [sample_with_temperature(logits, temp) for _ in range(1000)]
    counts = np.bincount(samples, minlength=len(logits))
    print(f"T={temp:.1f}: {counts / 1000}")
```

### 7.2 PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from temperature-scaled distribution.
    
    Args:
        logits: Tensor of shape (vocab_size,)
        temperature: Scaling factor
    
    Returns:
        Sampled token index
    """
    # Scale and apply Softmax
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Sample from categorical distribution
    return torch.multinomial(probs, num_samples=1)


# Example with batch processing
batch_logits = torch.randn(4, 50257)  # Batch of 4, vocab size 50257
temperatures = torch.tensor([0.5, 1.0, 1.5, 2.0]).unsqueeze(1)

# Apply different temperatures to each batch element
scaled_logits = batch_logits / temperatures
probs = F.softmax(scaled_logits, dim=-1)
samples = torch.multinomial(probs, num_samples=1)
```

### 7.3 Hugging Face Transformers Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with different temperatures
for temp in [0.3, 0.7, 1.0, 1.5]:
    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=temp,
        do_sample=True,  # Enable sampling (required for temperature)
        num_return_sequences=1
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTemperature {temp}:")
    print(text)
```

---

## 8. Advanced Topics

### 8.1 Adaptive Temperature Scheduling

Dynamically adjust temperature during generation:

```python
def adaptive_temperature(step: int, max_steps: int, 
                        start_temp: float = 1.5, 
                        end_temp: float = 0.5) -> float:
    """
    Linearly decrease temperature from start to end.
    Start creative, end deterministic.
    """
    return start_temp - (start_temp - end_temp) * (step / max_steps)
```

**Use Case:** Start with high creativity for ideation, then focus on coherent completion.

### 8.2 Token-Specific Temperature

Apply different temperatures to different token types:

```python
def token_specific_temperature(logits: np.ndarray, 
                               token_types: np.ndarray,
                               temp_map: dict) -> np.ndarray:
    """
    Apply different temperatures based on token type.
    
    Args:
        logits: Raw scores
        token_types: Array indicating type of each token
        temp_map: Dictionary mapping types to temperatures
    """
    scaled_logits = np.zeros_like(logits)
    for token_type, temp in temp_map.items():
        mask = (token_types == token_type)
        scaled_logits[mask] = logits[mask] / temp
    
    return softmax(scaled_logits)
```

**Example:** Use low temperature for function names, high temperature for variable names.

### 8.3 Temperature Calibration

Find optimal temperature for a specific task:

```python
def calibrate_temperature(model, validation_data, 
                         temp_range=(0.1, 2.0), 
                         num_trials=20):
    """
    Grid search for optimal temperature on validation set.
    """
    best_temp = 1.0
    best_score = -float('inf')
    
    for temp in np.linspace(*temp_range, num_trials):
        score = evaluate_model(model, validation_data, temperature=temp)
        if score > best_score:
            best_score = score
            best_temp = temp
    
    return best_temp, best_score
```

---

## 9. Common Pitfalls and Debugging

### Pitfall 1: Temperature = 0

**Problem:** Division by zero in $z_i / T$

**Solution:** Use a small epsilon or implement special case for greedy decoding:

```python
if temperature < 1e-8:
    return np.argmax(logits)  # Greedy decoding
else:
    return sample_with_temperature(logits, temperature)
```

### Pitfall 2: Negative Temperature

**Problem:** Inverts probability distribution (high logits → low probabilities)

**Solution:** Always validate $T > 0$

### Pitfall 3: Ignoring Top-K/Top-P

**Problem:** High temperature with full vocabulary → nonsense

**Solution:** Combine temperature with nucleus sampling (see Lesson 4)

### Pitfall 4: Inconsistent Temperature Across Layers

**Problem:** Applying temperature to attention and output differently

**Solution:** Document and standardise where temperature is applied

---

## 10. Exercises and Practice Problems

### Exercise 1: Manual Calculation

**Given:** Logits $\mathbf{z} = [0.5, 1.5, 2.5]$

**Tasks:**
1. Compute probabilities for $T = 0.5$
2. Compute probabilities for $T = 2.0$
3. Calculate the entropy for each case
4. Verify that entropy increases with temperature

### Exercise 2: Implementation

**Task:** Implement a function that generates text with temperature annealing:
- Start at $T = 1.5$ for the first 10 tokens
- Linearly decrease to $T = 0.5$ by token 50
- Maintain $T = 0.5$ thereafter

### Exercise 3: Empirical Analysis

**Task:** Using a pre-trained model (e.g., GPT-2):
1. Generate 100 completions for the same prompt at $T \in \{0.3, 0.7, 1.0, 1.5\}$
2. Measure diversity using unique n-gram counts
3. Measure coherence using perplexity
4. Plot diversity vs. coherence for each temperature

### Exercise 4: Mathematical Proof

**Task:** Prove that for any logits $\mathbf{z}$ and temperatures $T_1 < T_2$:

$$H(P_{T_2}) \geq H(P_{T_1})$$

Where $H$ is Shannon entropy and $P_T$ is the temperature-scaled distribution.

---

## 11. Summary and Key Takeaways

1. **Physical Origin:** Temperature scaling derives from statistical mechanics and thermodynamics

2. **Mathematical Effect:** Temperature inversely scales logits before Softmax, controlling distribution sharpness

3. **Practical Impact:**
   - Low $T$ (0.0-0.3): Deterministic, focused, safe
   - Medium $T$ (0.7-1.0): Balanced, natural
   - High $T$ (1.5-2.0): Creative, diverse, risky

4. **Industry Standards:** Most models default to $T = 1.0$, with recommended ranges of [0.0, 2.0]

5. **Historical Evolution:** From Boltzmann machines (1986) → Knowledge distillation (2015) → Modern LLMs (2017-present)

6. **Best Practices:**
   - Validate $T > 0$
   - Combine with top-k/top-p for better control
   - Calibrate on validation data for specific tasks
   - Consider adaptive scheduling for long generations

---

## 12. Further Reading

### Foundational Papers
1. Hinton, G. E., & Sejnowski, T. J. (1986). "Learning and relearning in Boltzmann machines"
2. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network"
3. Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). "A Learning Algorithm for Boltzmann Machines"

### Modern Applications
1. Holtzman, A., et al. (2019). "The Curious Case of Neural Text Degeneration" (discusses temperature with nucleus sampling)
2. Keskar, N. S., et al. (2019). "CTRL: A Conditional Transformer Language Model for Controllable Generation"

### Textbooks
1. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Chapter 10.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 20.10.

---

*End of Lesson 2*
