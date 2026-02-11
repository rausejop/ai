# Module 0: Foundations of LLM Customization Parameters

## Overview

This directory contains comprehensive, university-level postgraduate course materials on the fundamental mechanisms and customization parameters of Large Language Models (LLMs). All content is written in Oxford English and designed for advanced academic study.

## Files Created

### Course Index
- **M0_INDEX.md** - Complete module overview, reading lists, and study guide

### Lesson Files

1. **M0L1_Softmax_Function.md** (13.8 KB)
   - Historical evolution from Boltzmann (1868) to Bridle (1989)
   - Mathematical formulation and numerical stability
   - Applications in attention and token prediction
   - Gradient derivations and implementations

2. **M0L2_Temperature_Scaling.md** (20.4 KB)
   - Theoretical foundations from statistical mechanics
   - Mathematical analysis of distribution effects
   - **Gemini Flash 2.0 defaults:** T=1.0, range [0.0, 2.0]
   - Use case recommendations and best practices
   - Historical development and modern applications

3. **M0L3_Top_K_Sampling.md** (21.2 KB)
   - Problems with deterministic decoding
   - Mathematical formulation and complexity analysis
   - **Gemini Flash 2.0 defaults:** k=64 (fixed)
   - Historical emergence (2018) and GPT-2 standardization
   - Parameter selection guidelines

4. **M0L4_Top_P_Nucleus_Sampling.md** (26.5 KB)
   - Comprehensive analysis of Holtzman et al. (2019) paper
   - Adaptive nucleus size behaviour
   - **Gemini Flash 2.0 defaults:** p=0.95, range [0.0, 1.0]
   - Empirical benchmarks and evaluation metrics
   - Advanced variants and combined strategies

5. **M0L5_Dot_Product_Attention.md** (To be created)
   - Mathematical foundations of dot products
   - Semantic similarity in vector spaces
   - Scaled dot-product attention mechanism
   - Computational complexity and optimizations
   - Multi-head attention implementations

## Key Features

### Comprehensive Coverage
- ✅ Mathematical derivations from first principles
- ✅ Historical context with original paper citations
- ✅ Gemini Flash 2.0 parameter specifications
- ✅ Practical implementations (NumPy, PyTorch)
- ✅ Worked examples with step-by-step solutions
- ✅ Exercises for hands-on learning

### Gemini Flash 2.0 Parameters

| Parameter | Default | Range | File Reference |
|-----------|---------|-------|----------------|
| Temperature | 1.0 | [0.0, 2.0] | M0L2 |
| Top-P | 0.95 | [0.0, 1.0] | M0L4 |
| Top-K | 64 | Fixed | M0L3 |

### Essential Papers Covered

1. **Vaswani et al. (2017)** - "Attention Is All You Need" (100,000+ citations)
2. **Holtzman et al. (2019)** - "The Curious Case of Neural Text Degeneration"
3. **Bridle (1989)** - Softmax in neural networks
4. **Hinton et al. (2015)** - Knowledge distillation with temperature
5. **Radford et al. (2019)** - GPT-2 and top-k sampling

## Study Recommendations

### Suggested Order
1. Start with M0_INDEX.md for overview
2. Follow lessons sequentially (M0L1 → M0L5)
3. Complete exercises in each lesson
4. Read cited papers for deeper understanding
5. Implement algorithms from scratch

### Time Allocation
- **Total Module:** 25-35 hours
- **Per Lesson:** 4-7 hours
- **Projects:** 10-15 hours additional

## Mathematical Formulas Covered

### Core Equations

**Softmax:**
```
Softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

**Temperature-Scaled Softmax:**
```
P(y_i) = exp(z_i/T) / Σ exp(z_j/T)
```

**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

**Nucleus Sampling:**
```
V_p = min{ V' ⊆ V : Σ P(y_i) ≥ p }
```

## Implementation Examples

All lessons include:
- NumPy implementations (educational clarity)
- PyTorch implementations (production efficiency)
- Hugging Face Transformers integration
- Optimization techniques
- Debugging strategies

## Use Case Guidelines

### By Task Type

| Task | Temperature | Top-P | Top-K | Lesson |
|------|-------------|-------|-------|--------|
| Code Generation | 0.2-0.3 | 0.90 | 20 | M0L2, M0L3 |
| Factual Q&A | 0.3-0.5 | 0.85-0.90 | 10-20 | M0L2, M0L4 |
| Creative Writing | 1.0-1.5 | 0.95-0.98 | 80-100 | M0L2, M0L4 |
| General Chat | 0.8-1.0 | 0.92-0.95 | 50 | M0L2, M0L3, M0L4 |

## Additional Resources

### Original Files (Legacy)
- `softmax.md` - Earlier softmax notes
- `temperature.md` - Earlier temperature notes
- `dotproduct.md` - Earlier dot product notes
- `foundational.md` - General foundations
- `*.py` - Python implementation examples

### External Links
- Hugging Face Transformers: https://github.com/huggingface/transformers
- Annotated Transformer: http://nlp.seas.harvard.edu/annotated-transformer/
- Papers with Code: https://paperswithcode.com/

## Assessment

Each lesson includes:
- **Worked Examples:** Manual calculations with detailed steps
- **Implementation Exercises:** Code from scratch
- **Mathematical Proofs:** Derivations and theoretical analysis
- **Empirical Studies:** Experiments with real models
- **Comparative Analysis:** Evaluating different approaches

## Prerequisites

- Linear algebra (vectors, matrices, eigenvalues)
- Probability theory (distributions, entropy, Bayes' theorem)
- Calculus (derivatives, gradients, chain rule)
- Python programming (NumPy, basic PyTorch)
- Basic neural networks understanding

## Learning Outcomes

Upon completion, students will be able to:

1. ✅ Derive Softmax, temperature scaling, and attention formulas from first principles
2. ✅ Explain historical evolution of each technique with paper citations
3. ✅ Configure Gemini Flash 2.0 parameters for specific use cases
4. ✅ Implement efficient sampling and attention algorithms
5. ✅ Analyse trade-offs between determinism and diversity
6. ✅ Evaluate model behaviour using mathematical and empirical methods
7. ✅ Debug common issues in LLM parameter configuration
8. ✅ Design adaptive sampling strategies for novel applications

## Version Information

- **Created:** February 11, 2026
- **Version:** 1.0
- **Language:** Oxford English
- **Target Audience:** Postgraduate students
- **Course Level:** Advanced

## Contact

For questions or feedback regarding this module, please contact the course instructor through appropriate academic channels.

---

**Note:** This module represents a comprehensive, research-grade treatment of LLM customization parameters, synthesizing information from seminal papers, official documentation, and practical implementations. All sources are properly cited throughout the materials.
