# Chapter 8.3: CoT: Eliciting Reasoning and Planning

## 1. The Need for Step-by-Step Reasoning
Standard prompting attempts to map an input direct to an output. However, for tasks requiring complex logic—such as mathematics, symbolic reasoning, or multi-step strategic planning—this approach frequently fails. Large Language Models often "rush" to a conclusion, making silly errors in intermediate steps. **Chain-of-Thought (CoT)** prompting resolves this by forcing the model to allocate more **"Computational Working Memory"** to the reasoning process.

## 2. Manual CoT Prompting
In **Manual CoT**, the developer provides few-shot examples where the correct answer is preceded by a human-like explanation of the logic. By observing these "demonstrations" of reasoning, the model's internal attention is trained to decompose the final user query into a sequence of logical sub-goals. This approach has been mathematically shown to significantly increase accuracy on difficult benchmarks like GSM8K (grade-school math).

## 3. Automatic CoT (Self-Consistency)
Simple CoT can sometimes lead a model down a "hallucinated" path of logic. **Self-Consistency** mitigates this by asking the model to generate multiple different reasoning paths (e.g., 10 separate outputs) for the same problem. A background script then analyzes the final answers and chooses the one that appears most frequently (**The Majority Vote**). This collective intelligence approach filters out "one-off" logical mistakes, providing a much higher reliability score for production systems.

## 4. Demo: CoT for Mathematical and Logical Tasks
By simply appending the phrase **"Let's think step by step"** to a prompt (**Zero-Shot CoT**), a model's performance on logic puzzles can jump by over 30%. This illustrates that the reasoning capability is latent within the model's weights and only needs the correct linguistic "Trigger" to be activated. This technique is now a standard requirement for building AI agents that perform autonomous data analysis or code debugging.

## 5. Advanced CoT Variants (Tree-of-Thought, etc.)
The current frontier of reasoning is the **Tree-of-Thought (ToT)** framework. Unlike linear CoT, ToT allows the model to explore multiple "Branches" of thought simultaneously. If a branch reaches a logical dead-end, the model can "Backtrack" to a previous node and try a different path. This mimics high-level human problem-solving in complex domains like software architecture design or creative plot development, where a single linear path is rarely optimal.
By mastering these elicitation techniques, we move from "Prompting as a Question" to "Prompting as a Multi-step Cognitive Process."
