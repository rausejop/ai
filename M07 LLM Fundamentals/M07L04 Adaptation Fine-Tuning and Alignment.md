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
As established in Module 09, full fine-tuning of massive models is often prohibitively expensive. **PEFT** methodologies like **LoRA** (Low-Rank Adaptation) allow organizations to specialized their models by training less than 0.1% of the weights. This allows for rapid, domain-specific customization on consumer-grade hardware while preserving the general reasoning capabilities inherited from the massive pre-training phase. Through these integrated strategies, we move from general AI to specialized industrial solutions.
