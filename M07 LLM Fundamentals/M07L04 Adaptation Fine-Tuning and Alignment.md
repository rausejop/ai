# Chapter 7.4: Adaptation: Fine-Tuning and Alignment

## 1. Pre-training vs. Fine-Tuning: A Comparison
While **Pre-training** provides the model with general "Knowledge," it does not provide "Behavior." A base model trained on the internet is essentially a document-completer. If asked a question, it might respond with another question or a fictional scenario. **Fine-tuning** is the technical adaptation phase that transforms this raw statistical power into a functional and controllable assistant.

## 2. Instruction Tuning (Supervised Fine-Tuning - SFT)
**SFT** is the process of training the model on a curated dataset of **(Instruction, Response)** pairs. For example: *"User: Summarize this report. Assistant: [Concise Summary]"*. By shown 10,000 to 50,000 of these expert examples, the model learns the "format" of being a helpful assistant, transforming its stochastic completions into structured, goal-oriented responses.

## 3. Alignment: The Role of Human Feedback (RLHF)
To capture the subtle nuances of human preference (e.g., "be polite but authoritative"), models undergo **RLHF** (Reinforcement Learning from Human Feedback). 
- **Reward Modeling**: Humans rank multiple model responses. A separate "Reward Model" is trained to predict these rankings. 
- **PPO Optimization**: The main LLM is then fine-tuned to maximize its score from the Reward Model. 
This iterative cycle aligns the model with human ethical constraints and safety guidelines, ensuring it is "Helpful, Honest, and Harmless."

## 4. Parameter-Efficient Fine-Tuning (PEFT) Overview
As established in Module 09, full fine-tuning of massive models is often prohibitively expensive. **PEFT** methodologies like **LoRA** (Low-Rank Adaptation) allow organizations to specialized their models by training less than 0.1% of the weights. This allows for rapid, domain-specific customization on consumer-grade hardware while preserving the general reasoning capabilities inherited from the massive pre-training phase. 

## 📊 Visual Resources and Diagrams

- **The InstructGPT Pipeline**: A definitive 3-stage visual for SFT, Reward Modeling, and RLHF.
    - [Source: Ouyang et al. (2022) - Training language models to follow instructions (Fig 2)](https://arxiv.org/pdf/2203.02155.pdf)
- **Reward Model Distribution**: An infographic showing how the model learns to shift its output toward "Preferred" responses.
    - [Source: OpenAI - Learning from Human Feedback (Blog Image)](https://openai.com/wp-content/uploads/2023/04/rlhf-reward-model.png)

## 🐍 Technical Implementation (Python 3.14.2)

A simulation of a **Reward Model Scorer** logic used in RLHF on Windows.

```python
import torch
import torch.nn as nn

class RewardModelScorer(nn.Module):
    """
    Simulates a Reward Model that 'judges' LLM outputs.
    Compatible with Python 3.14.2.
    """
    def __init__(self, hidden_dim=764):
        super().__init__()
        # A simple regression head on top of the LLM embedding
        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: torch.Tensor):
        # Maps the latent state to a scalar 'Human Preference Score'
        score = torch.sigmoid(self.regression_head(embedding))
        return score

if __name__ == "__main__":
    judge = RewardModelScorer()
    
    # Simulating the latent vectors for a 'Polite' vs 'Rude' response
    polite_response_emb = torch.randn(1, 764) + 0.5
    rude_response_emb = torch.randn(1, 764) - 0.5
    
    polite_quality = judge(polite_response_emb)
    rude_quality = judge(rude_response_emb)
    
    print(f"Decision Boundary: Polite={polite_quality.item():.2f}, Rude={rude_quality.item():.2f}")
    if polite_quality > rude_quality:
        print("RLHF Status: Weights updated toward the polite vector.")
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Ouyang et al. (2022)**: *"Training language models to follow instructions with human feedback"*. (The InstructGPT paper).
    - [Link to ArXiv](https://arxiv.org/abs/2203.02155)
- **Rafailov et al. (2023)**: *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"*. (DPO - the modern alternative to RLHF).
    - [Link to ArXiv](https://arxiv.org/abs/2305.18290)

### Frontier News and Updates (2025-2026)
- **Anthropic Tech Blog (January 2026)**: Introduction of *Constitutional-DPO*—aligning models using only 1,000 "Golden Principles" instead of manual human ranking.
- **NVIDIA AI News**: "The Alignment Bottleneck"—New software for accelerating RLHF cycles on large H200 clusters.
- **Grok (xAI) Tech Blog**: Discussion on "Adversarial Alignment"—training models to stay safe even when a user is actively trying to provoke a harmful response.
