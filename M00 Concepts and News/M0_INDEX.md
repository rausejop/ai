# Module 0: Foundations of Large Language Model Customization Parameters

## Course Overview

This module provides comprehensive, university-level postgraduate education on the fundamental mechanisms and customization parameters of Large Language Models (LLMs). Each lesson combines rigorous mathematical foundations, historical context, practical implementations, and real-world applications.

**Target Audience:** Postgraduate students in Computer Science, AI/ML, Computational Linguistics, or related fields

**Prerequisites:** 
- Linear algebra (vectors, matrices, dot products)
- Probability theory (probability distributions, entropy)
- Calculus (derivatives, gradients)
- Python programming
- Basic understanding of neural networks

**Learning Outcomes:**

By completing this module, students will be able to:
1. Derive and implement core LLM mathematical operations from first principles
2. Explain the historical evolution of key techniques from seminal papers
3. Configure LLM parameters for specific use cases with theoretical justification
4. Analyse trade-offs between different sampling strategies
5. Implement efficient algorithms for attention and sampling mechanisms
6. Evaluate model behaviour using mathematical and empirical analysis

---

## Module Structure

### Lesson 1: The Softmax Function
**File:** `M0L1_Softmax_Function.md`

**Topics Covered:**
- Historical evolution from Boltzmann distributions (1868) to neural networks (1989)
- Mathematical formulation and properties
- Numerical stability considerations
- Gradient derivation for backpropagation
- Applications in attention mechanisms and token prediction
- Computational complexity analysis
- Practical implementations (NumPy, PyTorch)

**Key Papers:**
- Boltzmann, L. (1868) - Statistical mechanics foundations
- Luce, R. D. (1959) - Choice axiom
- Bridle, J. S. (1989) - Neural network formulation
- Vaswani, A., et al. (2017) - Transformer applications

**Estimated Study Time:** 4-6 hours

---

### Lesson 2: Temperature Scaling
**File:** `M0L2_Temperature_Scaling.md`

**Topics Covered:**
- Theoretical foundations from statistical mechanics
- Mathematical analysis of temperature effects on probability distributions
- Entropy analysis and limiting behaviour
- **Gemini Flash 2.0 defaults:** T=1.0, range [0.0, 2.0]
- Use case recommendations by temperature range
- Historical development from Boltzmann machines to modern LLMs
- Implementation with adaptive scheduling
- Common pitfalls and debugging strategies

**Key Papers:**
- Hinton, G. E., & Sejnowski, T. J. (1986) - Boltzmann machines
- Hinton, G., et al. (2015) - Knowledge distillation
- Vaswani, A., et al. (2017) - Transformer standardization

**Estimated Study Time:** 4-6 hours

---

### Lesson 3: Top-K Sampling
**File:** `M0L3_Top_K_Sampling.md`

**Topics Covered:**
- Problems with deterministic decoding (greedy, beam search)
- Mathematical formulation of top-k sampling
- Historical emergence (2018) and standardization in GPT-2 (2019)
- **Gemini Flash 2.0 defaults:** k=64 (fixed)
- Computational complexity and optimization strategies
- Entropy analysis and probability distribution effects
- Parameter selection by use case
- Limitations and comparison with nucleus sampling

**Key Papers:**
- Fan, A., et al. (2018) - Early applications
- Radford, A., et al. (2019) - GPT-2 popularization
- Holtzman, A., et al. (2019) - Critical analysis

**Estimated Study Time:** 4-5 hours

---

### Lesson 4: Top-P (Nucleus) Sampling
**File:** `M0L4_Top_P_Nucleus_Sampling.md`

**Topics Covered:**
- Motivation: adaptive sampling vs. fixed-k
- Mathematical formulation of nucleus sampling
- **Comprehensive analysis of seminal paper:** Holtzman et al. (2019)
- **Gemini Flash 2.0 defaults:** p=0.95, range [0.0, 1.0]
- Adaptive nucleus size behaviour
- Empirical benchmarks and evaluation metrics
- Combined top-k and top-p strategies
- Advanced variants (tail-free sampling, typical sampling)

**Key Papers:**
- **Holtzman, A., et al. (2019)** - "The Curious Case of Neural Text Degeneration" ⭐ Essential
- Meister, C., et al. (2022) - Typical decoding
- Basu, S., et al. (2020) - Mirostat

**Estimated Study Time:** 5-7 hours

---

### Lesson 5: Dot Product and Scaled Dot-Product Attention
**File:** `M0L5_Dot_Product_Attention.md`

**Topics Covered:**
- Mathematical foundations: algebraic and geometric formulations
- Dot product in semantic vector spaces
- Cosine similarity and distance metrics
- Query-Key-Value paradigm
- **Scaled dot-product attention formula:** Why $\sqrt{d_k}$?
- Gradient flow analysis
- Computational complexity: $O(n^2 d)$ bottleneck
- Modern optimizations (Flash Attention, sparse attention)
- Multi-head attention implementation

**Key Papers:**
- Bahdanau, D., et al. (2014) - Attention mechanism origins
- **Vaswani, A., et al. (2017)** - "Attention Is All You Need" ⭐ Essential (100,000+ citations)
- Dao, T., et al. (2022) - Flash Attention
- Choromanski, K., et al. (2020) - Performers

**Estimated Study Time:** 6-8 hours

---

## Gemini Flash 2.0 Parameter Reference

### Default Configuration

| Parameter | Default Value | Range | Description |
|-----------|---------------|-------|-------------|
| **Temperature** | 1.0 | [0.0, 2.0] | Controls randomness in sampling |
| **Top-P** | 0.95 | [0.0, 1.0] | Nucleus sampling threshold |
| **Top-K** | 64 | Fixed | Number of top candidates (combined with top-p) |

**Source:** Google AI Vertex AI Documentation (2024)

### Recommended Configurations by Task

| Task Type | Temperature | Top-P | Top-K | Rationale |
|-----------|-------------|-------|-------|-----------|
| **Code Generation** | 0.2-0.3 | 0.90 | 20 | High precision, limited valid syntax |
| **Factual Q&A** | 0.3-0.5 | 0.85-0.90 | 10-20 | Prefer confident answers |
| **Technical Documentation** | 0.5-0.7 | 0.90-0.92 | 30 | Coherent, professional |
| **General Chat** | 0.8-1.0 | 0.92-0.95 | 50 | Natural, conversational |
| **Creative Writing** | 1.0-1.5 | 0.95-0.98 | 80-100 | Diverse vocabulary |
| **Brainstorming** | 1.2-1.8 | 0.98 | 100 | Maximum creativity |

---

## Mathematical Formulas Quick Reference

### Softmax
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

### Temperature-Scaled Softmax
$$P(y_i) = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}}$$

### Top-K Sampling
$$P_{\text{top-k}}(y_i) = \begin{cases}
\frac{P(y_i)}{\sum_{j \in V_k} P(y_j)} & \text{if } i \in V_k \\
0 & \text{otherwise}
\end{cases}$$

### Nucleus (Top-P) Sampling
$$V_p = \min \left\{ V' \subseteq V : \sum_{i \in V'} P(y_i) \geq p \right\}$$

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Dot Product
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$

### Cosine Similarity
$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

---

## Essential Reading List

### Foundational Papers (Must Read)

1. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*. 
   - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   - 100,000+ citations - Most influential paper in modern NLP

2. **Holtzman, A., et al. (2019).** "The Curious Case of Neural Text Degeneration." *ICLR 2020*.
   - [https://arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751)
   - Introduced nucleus sampling, critical analysis of decoding methods

3. **Bridle, J. S. (1989).** "Probabilistic Interpretation of Feedforward Classification Network Outputs."
   - [https://link.springer.com/chapter/10.1007/978-3-642-75100-4_13](https://link.springer.com/chapter/10.1007/978-3-642-75100-4_13)
   - Formal introduction of Softmax to neural networks

### Historical Context

4. Boltzmann, L. (1868). "Studien über das Gleichgewicht der lebendigen Kraft"
   - Origins of exponential probability distributions

5. Luce, R. D. (1959). "Individual Choice Behavior: A Theoretical Analysis"
   - [https://psycnet.apa.org/record/1959-08955-000](https://psycnet.apa.org/record/1959-08955-000)
   - Choice axiom foundations

6. Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
   - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
   - First attention mechanism in NMT

### Modern Developments

7. Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network"
   - [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
   - Temperature in knowledge distillation

8. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
   - Popularized top-k sampling

9. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
   - [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
   - Modern attention optimization

### Textbooks

10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
    - Chapters 6 (Deep Feedforward Networks), 10 (Sequence Modeling)

11. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.)
    - Chapter 10 (Transformers and Large Language Models)

12. Tunstall, L., et al. (2022). *Natural Language Processing with Transformers*. O'Reilly.
    - Practical implementations and applications

---

## Practical Implementation Resources

### Code Repositories

1. **Hugging Face Transformers**
   - [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
   - Industry-standard implementations

2. **Annotated Transformer**
   - [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)
   - Line-by-line explanation of Transformer implementation

3. **Flash Attention**
   - [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
   - Optimized attention kernels

### Online Courses

1. Stanford CS224N: Natural Language Processing with Deep Learning
2. Fast.ai: Practical Deep Learning for Coders
3. DeepLearning.AI: Natural Language Processing Specialization

---

## Assessment and Exercises

Each lesson includes:
- **Worked Examples:** Step-by-step calculations with detailed explanations
- **Implementation Exercises:** Coding tasks from scratch and with libraries
- **Mathematical Proofs:** Derivations and theoretical analysis
- **Empirical Studies:** Experiments with real models and datasets
- **Comparative Analysis:** Evaluating different approaches

### Suggested Projects

1. **Parameter Tuning Study:** Systematically evaluate temperature, top-k, and top-p combinations on a specific task

2. **Sampling Algorithm Implementation:** Build a complete sampling library with all methods covered

3. **Attention Visualization Tool:** Create interactive visualizations of attention patterns

4. **Performance Benchmarking:** Compare computational efficiency of different attention implementations

5. **Adaptive Sampling System:** Design and evaluate adaptive parameter selection based on context

---

## Study Recommendations

### Week 1-2: Foundations
- Lesson 1: Softmax Function
- Lesson 2: Temperature Scaling
- Review linear algebra and probability theory as needed

### Week 3-4: Sampling Strategies
- Lesson 3: Top-K Sampling
- Lesson 4: Top-P Sampling
- Implement and compare both methods

### Week 5-6: Attention Mechanisms
- Lesson 5: Dot Product and Attention
- Study Transformer architecture in depth
- Implement multi-head attention

### Week 7-8: Integration and Projects
- Complete all exercises
- Work on suggested projects
- Read additional papers from reference list

---

## Additional Resources

### Online Tools

1. **Tensor Playground:** Visualize matrix operations
2. **Attention Visualizer:** Interactive attention heatmaps
3. **LLM Playground:** Experiment with different parameters

### Datasets for Practice

1. **WikiText-103:** Language modeling
2. **SQuAD:** Question answering
3. **GLUE Benchmark:** Various NLP tasks
4. **The Pile:** Large-scale diverse text

### Community and Support

1. **Hugging Face Forums:** [https://discuss.huggingface.co/](https://discuss.huggingface.co/)
2. **r/MachineLearning:** Reddit community
3. **Papers with Code:** [https://paperswithcode.com/](https://paperswithcode.com/)

---

## Glossary

- **Attention:** Mechanism for computing weighted combinations of values based on query-key similarity
- **Beam Search:** Decoding strategy maintaining top-k hypotheses at each step
- **Cosine Similarity:** Normalized dot product measuring directional similarity
- **Dot Product:** Sum of element-wise products of two vectors
- **Embedding:** Dense vector representation of discrete tokens
- **Entropy:** Measure of uncertainty in a probability distribution
- **Greedy Decoding:** Always selecting the most probable token
- **Logits:** Raw, unnormalized scores before softmax
- **Nucleus:** Set of tokens whose cumulative probability exceeds threshold p
- **Perplexity:** Exponential of cross-entropy, measures model uncertainty
- **Query (Q):** Vector representing what information to look for
- **Key (K):** Vector representing what information is available
- **Value (V):** Vector containing the actual information
- **Sampling:** Stochastically selecting tokens from a probability distribution
- **Softmax:** Function converting logits to probability distribution
- **Temperature:** Scaling parameter controlling randomness in sampling
- **Top-K:** Sampling from k most probable tokens
- **Top-P:** Sampling from smallest set covering p probability mass
- **Transformer:** Neural architecture based on self-attention mechanisms

---

## Version History

- **Version 1.0** (February 2026): Initial comprehensive module creation
  - 5 detailed lessons covering core LLM customization parameters
  - Gemini Flash 2.0 parameter specifications
  - Historical context from seminal papers
  - Mathematical derivations and proofs
  - Practical implementations in NumPy and PyTorch
  - Extensive worked examples and exercises

---

## Contact and Feedback

For questions, corrections, or suggestions regarding this module, please refer to the course instructor or submit issues through the appropriate academic channels.

---

## License and Attribution

This educational material synthesizes information from publicly available research papers, documentation, and textbooks. All original sources are cited appropriately. Code examples are provided for educational purposes.

**Key Sources:**
- Google AI Vertex AI Documentation
- Original research papers (cited throughout)
- PyTorch and Hugging Face documentation
- Academic textbooks (see Essential Reading List)

---

*Module 0: Foundations of LLM Customization Parameters*  
*Postgraduate Course in Large Language Model Engineering*  
*Last Updated: February 2026*
