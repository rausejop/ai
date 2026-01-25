# Chapter 9.4: RLHF: Aligning Models with Human Feedback

## 1. The Alignment Problem in LLMs
A model fine-tuned on instructions (SFT) is capable of following commands, but it often possesses "Stochastic Biases" or "Unsafe Behaviors" inherited from the raw internet. It may be overly redundant, sarcastic, or provide harmful instructions. **Alignment** is the technical process of ensuring that the model's stochastic outputs match the subtle preferences of human society.

## 2. Reinforcement Learning from Human Feedback
**RLHF** (Reinforcement Learning from Human Feedback) is the definitive protocol for large-scale model alignment. It transforms human preference into a mathematical reward signal that the model can optimize against. This three-stage process is what transformed the original generative models into the helpful assistants known today.

## 3. Step 1: Supervised Fine-Tuning (SFT)
The first step is building the **Instruction Base**. The model is fine-tuned on $10,000 \dots 50,000$ high-quality, human-written (Prompt, Response) pairs. This training provides the model with a strong behavioral starting point, ensuring it understands the basic structure of a helpful conversation.

## 4. Step 2: Reward Model Training
Human values are too complex to be captured by a simple mathematical loss function. Instead, we train a second "Judge" model:
- **Process**: Humans rank multiple model responses.
- **Training**: A smaller network, the **Reward Model**, is trained to predict these rankings. It becomes a digital proxy for human preference.

## 5. Step 3: PPO (Proximal Policy Optimization)
In the final stage, the main LLM is fine-tuned using Reinforcement Learning. The model generates responses, and the Reward Model "scores" them. The **PPO** algorithm then updates the weights of the LLM to maximize this reward score while ensuring the model doesn't drift too far from its original pre-trained quality.

## 📊 Visual Resources and Diagrams

- **The RLHF Loop Breakdown**: A visual showing SFT $\rightarrow$ Reward Model $\rightarrow$ RLHF (PPO).
    - [Source: Ouyang et al. (2022) - InstructGPT (Fig 2)](https://arxiv.org/pdf/2203.02155.pdf)
- **DPO vs. PPO Pipeline**: An infographic by Hugging Face showing the simplified Direct Preference Optimization logic.
    - [Source: TRL Blog - Direct Preference Optimization](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/dpo/dpo_diagram.png)

## 🐍 Technical Implementation (Python 3.14.2)

A simulation of a **Direct Preference Optimization (DPO)** loss calculation on Windows.

```python
import torch
import torch.nn.functional as F

def compute_dpo_loss(policy_logits, reference_logits, beta=0.1):
    """
    Simulates the DPO loss for aligning models without a reward model.
    Matches the policy model against a frozen reference model.
    Compatible with Python 3.14.2.
    """
    # 1. Compute the log-ratio of the policy model probabilities
    # policy_logits: [2, vocab] -> chosen vs rejected
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(reference_logits, dim=-1)
    
    # 2. Extract specific probabilities for 'chosen' and 'rejected' completions
    # (Simplified for demonstration)
    prob_chosen = policy_log_probs[0].mean() - ref_log_probs[0].mean()
    prob_rejected = policy_log_probs[1].mean() - ref_log_probs[1].mean()
    
    # 3. Calculate DPO Loss (Implicitly optimizes for preference)
    loss = -F.logsigmoid(beta * (prob_chosen - prob_rejected))
    
    return loss.item()

if __name__ == "__main__":
    # Mock logits for a 'Better' vs 'Worse' response
    mock_policy = torch.randn(2, 50257) 
    mock_ref = torch.randn(2, 50257)
    
    l = compute_dpo_loss(mock_policy, mock_ref)
    print(f"Alignment Signal: DPO-Loss={l:.4f}")
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Ouyang et al. (2022)**: *"Training language models to follow instructions with human feedback"*.
    - [Link to ArXiv](https://arxiv.org/abs/2203.02155)
- **Rafailov et al. (2023)**: *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"*.
    - [Link to ArXiv](https://arxiv.org/abs/2305.18290)

### Frontier News and Updates (2025-2026)
- **OpenAI (September 2025)**: Introduction of *o1-Preferred*, an alignment strategy where models "reason" about their own preferences before updating weights.
- **NVIDIA AI Blog**: "The Throughput of DPO"—How Blackwell systems execute 1,000 preference updates per second.
- **Anthropic Tech Blog**: "Constitutional-RLHF"—Using a written 'Constitution' to replace thousands of hours of manual human ranking.
