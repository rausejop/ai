# Chapter 9.4: RLHF: Aligning Models with Human Feedback

## 1. The Alignment Problem in LLMs
A model fine-tuned on instructions (SFT) is capable of following commands, but it often possesses "Stochastic Biases" or "Unsafe Behaviors" inherited from the raw internet. It may be overly redundant, sarcastic, or provide harmful instructions. **Alignment** is the technical process of ensuring that the model's stochastic outputs match the subtle, complex, and often conflicting preferences of human society.

## 2. Reinforcement Learning from Human Feedback
**RLHF** (Reinforcement Learning from Human Feedback) is the definitive protocol for large-scale model alignment. It transforms human preference into a mathematical reward signal that the model can optimize against. This three-stage process is what transformed the original GPT-3 into the helpful assistant known as ChatGPT.

## 3. Step 1: Supervised Fine-Tuning (SFT)
The first step is building the **Instruction Base**. The model is fine-tuned on $10,000 \dots 50,000$ high-quality, human-written (Prompt, Response) pairs. This "Supervised" training provides the model with a strong behavioral starting point, ensuring it understands the basic structure of a helpful conversation.

## 4. Step 2: Reward Model Training
Human values are too complex to be captured by a simple mathematical loss function. Instead, we train a second "Judge" model:
- **Process**: Humans are shown multiple model responses to the same prompt and asked to rank them (e.g., "Response A is more polite than B").
- **Training**: A smaller neural network, the **Reward Model**, is trained to predict these human rankings. It becomes a digital proxy for human preference, capable of "scoring" any new model output.

## 5. Step 3: PPO (Proximal Policy Optimization)
In the final stage, the main LLM is fine-tuned using Reinforcement Learning. The model generates responses, and the Reward Model "scores" them. The **PPO** algorithm then updates the weights of the LLM to maximize this reward score while ensuring the model doesn't drift too far from its original pre-trained linguistic quality. Through millions of iterations, the model’s weights are "aligned" with human values, creating an assistant that is not just intelligent, but also helpful, honest, and harmless.
