# Module 0, Lesson 1: The Softmax Function - Mathematical Foundations and Applications in Large Language Models

## Learning Objectives

By the end of this lesson, students will be able to:
1. Derive the mathematical formulation of the Softmax function from first principles
2. Explain the historical evolution from Boltzmann distributions to modern neural networks
3. Implement Softmax computations with numerical stability considerations
4. Analyse the role of Softmax in attention mechanisms and token prediction
5. Evaluate the computational complexity and optimisation strategies for Softmax operations

---

## 1. Historical Context and Theoretical Foundations

### 1.1 The Boltzmann Distribution (1868)

The conceptual genesis of the Softmax function originates from **Ludwig Boltzmann's** work in statistical mechanics. Boltzmann sought to describe the probability distribution of particles across discrete energy states in a thermodynamic system.

**The Boltzmann Distribution:**

$$P(i) = \frac{e^{-E_i / kT}}{\sum_{j} e^{-E_j / kT}}$$

Where:
- $P(i)$ = Probability of the system being in state $i$
- $E_i$ = Energy of state $i$
- $k$ = Boltzmann constant ($1.380649 \times 10^{-23}$ J/K)
- $T$ = Absolute temperature (Kelvin)

**Physical Interpretation:** Systems naturally tend towards states of lower energy. The exponential term $e^{-E_i / kT}$ ensures that lower-energy states have higher probabilities. The denominator normalises these values to form a valid probability distribution.

**Seminal Reference:**
- Boltzmann, L. (1868). "Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten." *Wiener Berichte*, 58, 517-560.

### 1.2 Statistical Mechanics to Machine Learning (1959-1989)

**R. Duncan Luce (1959)** introduced the **Luce's Choice Axiom**, which formalised the concept of probabilistic choice behaviour. This work established the mathematical foundation for what would later become the Softmax function in neural networks.

**Luce's Choice Rule:**

$$P(i) = \frac{v_i}{\sum_{j} v_j}$$

Where $v_i$ represents the "value" or "utility" of option $i$.

**Seminal Reference:**
- Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis*. New York: Wiley. [https://psycnet.apa.org/record/1959-08955-000](https://psycnet.apa.org/record/1959-08955-000)

### 1.3 The Neural Network Revolution (1989)

**John S. Bridle (1989)** formally introduced the term "Softmax" and its application to neural networks for pattern classification. His work established Softmax as the standard output activation function for multi-class classification problems.

**Seminal Reference:**
- Bridle, J. S. (1989). "Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition." In *Neurocomputing* (pp. 227-236). Springer, Berlin, Heidelberg. [https://link.springer.com/chapter/10.1007/978-3-642-75100-4_13](https://link.springer.com/chapter/10.1007/978-3-642-75100-4_13)

---

## 2. Mathematical Formulation

### 2.1 Standard Softmax Function

Given a vector of real-valued scores (logits) $\mathbf{z} = [z_1, z_2, \ldots, z_n]$, the Softmax function transforms these into a probability distribution:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

**Properties:**
1. **Range:** $\text{Softmax}(z_i) \in (0, 1)$ for all $i$
2. **Normalisation:** $\sum_{i=1}^{n} \text{Softmax}(z_i) = 1$
3. **Monotonicity:** If $z_i > z_j$, then $\text{Softmax}(z_i) > \text{Softmax}(z_j)$
4. **Differentiability:** Softmax is continuously differentiable, enabling gradient-based optimisation

### 2.2 Numerical Stability Considerations

**The Problem:** Direct computation of $e^{z_i}$ can lead to numerical overflow when $z_i$ is large (e.g., $e^{1000} \approx \infty$ in floating-point arithmetic).

**The Solution:** Subtract the maximum logit value before exponentiation:

$$\text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{n} e^{z_j - \max(\mathbf{z})}}$$

**Proof of Equivalence:**

Let $c = \max(\mathbf{z})$. Then:

$$\frac{e^{z_i - c}}{\sum_{j} e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_{j} e^{z_j} \cdot e^{-c}} = \frac{e^{z_i} \cdot e^{-c}}{e^{-c} \sum_{j} e^{z_j}} = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

### 2.3 Gradient Derivation (Backpropagation)

The gradient of Softmax with respect to its inputs is essential for training neural networks.

**For $i = j$:**

$$\frac{\partial \text{Softmax}(z_i)}{\partial z_i} = \text{Softmax}(z_i) \cdot (1 - \text{Softmax}(z_i))$$

**For $i \neq j$:**

$$\frac{\partial \text{Softmax}(z_i)}{\partial z_j} = -\text{Softmax}(z_i) \cdot \text{Softmax}(z_j)$$

**Jacobian Matrix:**

$$J_{ij} = \frac{\partial \text{Softmax}(z_i)}{\partial z_j} = \text{Softmax}(z_i) \cdot (\delta_{ij} - \text{Softmax}(z_j))$$

Where $\delta_{ij}$ is the Kronecker delta (1 if $i=j$, 0 otherwise).

---

## 3. Applications in Large Language Models

### 3.1 Scaled Dot-Product Attention

In Transformer architectures (Vaswani et al., 2017), Softmax is applied to compute attention weights:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Query matrix
- $K$ = Key matrix
- $V$ = Value matrix
- $d_k$ = Dimension of key vectors
- $\sqrt{d_k}$ = Scaling factor to prevent gradient saturation

**Seminal Reference:**
- Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 30. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 3.2 Token Prediction (Output Layer)

During text generation, the final layer produces logits for each token in the vocabulary. Softmax converts these to probabilities:

$$P(\text{token}_i | \text{context}) = \frac{e^{z_i}}{\sum_{j=1}^{|V|} e^{z_j}}$$

Where $|V|$ is the vocabulary size (e.g., 32,000 for GPT-2, 256,000 for Gemini models).

---

## 4. Computational Complexity Analysis

### 4.1 Time Complexity

For a vector of length $n$:
1. **Exponentiation:** $O(n)$ operations
2. **Summation:** $O(n)$ operations
3. **Division:** $O(n)$ operations

**Total:** $O(n)$ per Softmax operation

### 4.2 Space Complexity

- **Input:** $O(n)$ for storing logits
- **Output:** $O(n)$ for storing probabilities
- **Intermediate:** $O(1)$ for the sum accumulator

**Total:** $O(n)$

### 4.3 Optimisation Strategies

1. **Vectorisation:** Use SIMD (Single Instruction, Multiple Data) operations
2. **Fused Kernels:** Combine exponentiation and normalisation in GPU kernels
3. **Approximations:** Use polynomial or lookup table approximations for $e^x$
4. **Sparse Softmax:** Only compute probabilities for top-k candidates

---

## 5. Practical Implementation

### 5.1 Python Implementation (NumPy)

```python
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable Softmax implementation.
    
    Args:
        logits: Array of shape (n,) or (batch_size, n)
    
    Returns:
        Probability distribution of same shape as input
    """
    # Subtract max for numerical stability
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    
    # Compute exponentials
    exp_logits = np.exp(shifted_logits)
    
    # Normalise
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


# Example usage
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probabilities}")
print(f"Sum: {np.sum(probabilities)}")
```

**Expected Output:**
```
Logits: [2.  1.  0.1]
Probabilities: [0.65900114 0.24243297 0.09856589]
Sum: 1.0
```

### 5.2 PyTorch Implementation

```python
import torch
import torch.nn.functional as F

# Using built-in function
logits = torch.tensor([2.0, 1.0, 0.1])
probabilities = F.softmax(logits, dim=0)

# Manual implementation
def manual_softmax(logits: torch.Tensor) -> torch.Tensor:
    exp_logits = torch.exp(logits - torch.max(logits))
    return exp_logits / torch.sum(exp_logits)
```

---

## 6. Worked Examples and Exercises

### Example 1: Basic Softmax Computation

**Given:** Logits $\mathbf{z} = [1.0, 2.0, 3.0]$

**Step 1:** Compute exponentials
- $e^{1.0} = 2.718$
- $e^{2.0} = 7.389$
- $e^{3.0} = 20.086$

**Step 2:** Sum exponentials
- $\sum e^{z_j} = 2.718 + 7.389 + 20.086 = 30.193$

**Step 3:** Normalise
- $P(z_1) = 2.718 / 30.193 = 0.090$
- $P(z_2) = 7.389 / 30.193 = 0.245$
- $P(z_3) = 20.086 / 30.193 = 0.665$

**Verification:** $0.090 + 0.245 + 0.665 = 1.000$ ✓

### Example 2: Effect of Scaling

**Given:** Logits $\mathbf{z} = [1.0, 2.0, 3.0]$

**Case A:** Standard Softmax
- Result: $[0.090, 0.245, 0.665]$

**Case B:** Multiply logits by 2
- Input: $[2.0, 4.0, 6.0]$
- Result: $[0.015, 0.117, 0.868]$

**Observation:** Scaling logits makes the distribution more peaked (concentrated on the maximum).

### Exercise 1: Numerical Stability

**Task:** Compute Softmax for $\mathbf{z} = [1000, 1001, 1002]$ using:
1. Direct method (will overflow)
2. Numerically stable method

**Solution:**

Direct method:
- $e^{1000} \approx \infty$ (overflow!)

Stable method:
- $\max(\mathbf{z}) = 1002$
- Shifted: $[-2, -1, 0]$
- $e^{-2} = 0.135$, $e^{-1} = 0.368$, $e^{0} = 1.000$
- Sum = $1.503$
- Result: $[0.090, 0.245, 0.665]$

### Exercise 2: Gradient Computation

**Task:** Compute the Jacobian matrix for Softmax with input $\mathbf{z} = [1, 2]$

**Solution:**

First, compute Softmax:
- $e^1 = 2.718$, $e^2 = 7.389$
- Sum = $10.107$
- $\text{Softmax}(\mathbf{z}) = [0.269, 0.731]$

Jacobian:
$$J = \begin{bmatrix}
0.269 \times (1 - 0.269) & -0.269 \times 0.731 \\
-0.731 \times 0.269 & 0.731 \times (1 - 0.731)
\end{bmatrix} = \begin{bmatrix}
0.197 & -0.197 \\
-0.197 & 0.197
\end{bmatrix}$$

---

## 7. Advanced Topics

### 7.1 Log-Softmax

For numerical stability in loss computation, we often use Log-Softmax:

$$\log(\text{Softmax}(z_i)) = z_i - \log\left(\sum_{j} e^{z_j}\right)$$

This is more stable than computing Softmax then taking the logarithm.

### 7.2 Gumbel-Softmax

A differentiable approximation to categorical sampling:

$$\text{Gumbel-Softmax}(z_i, \tau) = \frac{e^{(z_i + g_i)/\tau}}{\sum_{j} e^{(z_j + g_j)/\tau}}$$

Where $g_i \sim \text{Gumbel}(0, 1)$ and $\tau$ is a temperature parameter.

**Reference:**
- Jang, E., Gu, S., & Poole, B. (2016). "Categorical Reparameterization with Gumbel-Softmax." *arXiv preprint arXiv:1611.01144*.

### 7.3 Sparse Softmax

For large vocabularies, compute Softmax only over a subset:

$$\text{Sparse-Softmax}(z_i) = \begin{cases}
\frac{e^{z_i}}{\sum_{j \in S} e^{z_j}} & \text{if } i \in S \\
0 & \text{otherwise}
\end{cases}$$

Where $S$ is the set of top-k candidates.

---

## 8. Common Pitfalls and Best Practices

### Pitfalls
1. **Numerical Overflow:** Always subtract max before exponentiation
2. **Underflow in Gradients:** Use Log-Softmax for loss computation
3. **Incorrect Axis:** Ensure Softmax is applied along the correct dimension

### Best Practices
1. Use library implementations (PyTorch, TensorFlow) when possible
2. Verify normalisation: probabilities should sum to 1
3. Monitor for NaN/Inf values during training
4. Use mixed-precision training carefully (FP16 can cause underflow)

---

## 9. Summary and Key Takeaways

1. **Historical Evolution:** Softmax evolved from Boltzmann's statistical mechanics (1868) through Luce's choice theory (1959) to Bridle's neural network formulation (1989)

2. **Mathematical Properties:** Softmax is a differentiable normalisation function that converts logits to probabilities

3. **LLM Applications:** Critical for both attention mechanisms and token prediction

4. **Numerical Stability:** Always use the max-subtraction trick for stable computation

5. **Computational Efficiency:** $O(n)$ time complexity, amenable to GPU acceleration

---

## 10. Further Reading

### Foundational Papers
1. Bridle, J. S. (1989). "Probabilistic Interpretation of Feedforward Classification Network Outputs"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. Luce, R. D. (1959). "Individual Choice Behavior: A Theoretical Analysis"

### Modern Applications
1. Shazeer, N. (2020). "GLU Variants Improve Transformer" (Softmax alternatives)
2. Katharopoulos, A., et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

### Textbooks
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.2.2.3.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.3.4.

---

## Appendix A: Softmax in Different Programming Languages

### Julia
```julia
function softmax(x::Vector{Float64})
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end
```

### R
```r
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  return(exp_x / sum(exp_x))
}
```

### C++ (Eigen)
```cpp
#include <Eigen/Dense>

Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}
```

---

## Appendix B: Glossary

- **Logits:** Raw, unnormalised scores output by a neural network layer
- **Normalisation:** Scaling values to sum to 1
- **Numerical Stability:** Avoiding overflow/underflow in floating-point arithmetic
- **Jacobian:** Matrix of all first-order partial derivatives
- **Gradient Saturation:** When gradients become very small, slowing learning

---

*End of Lesson 1*
