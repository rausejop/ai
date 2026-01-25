# Chapter 8.3: CoT: Eliciting Reasoning and Planning

## 1. The Need for Step-by-Step Reasoning
Standard prompting attempts to map an input direct to an output. However, for tasks requiring complex logic—such as mathematics, symbolic reasoning, or multi-step strategic planning—this approach frequently fails. Large Language Models often "rush" to a conclusion, making silly errors in intermediate steps. **Chain-of-Thought (CoT)** prompting resolves this by forcing the model to allocate more **"Computational Working Memory"** to the reasoning process.

## 2. Manual CoT Prompting
In **Manual CoT**, the developer provides few-shot examples where the correct answer is preceded by a human-like explanation of the logic. By observing these "demonstrations" of reasoning, the model's internal attention is trained to decompose the final user query into a sequence of logical sub-goals.

## 3. Automatic CoT (Self-Consistency)
Simple CoT can sometimes lead a model down a "hallucinated" path of logic. **Self-Consistency** mitigates this by asking the model to generate multiple different reasoning paths (e.g., 10 separate outputs) for the same problem. A background script then analyzes the final answers and chooses the one that appears most frequently (**The Majority Vote**). This collective intelligence approach filters out "one-off" logical mistakes.

## 4. Demo: CoT for Mathematical and Logical Tasks
By simply appending the phrase **"Let's think step by step"** to a prompt (**Zero-Shot CoT**), a model's performance on logic puzzles can jump by over 30%. This illustrates that the reasoning capability is latent within the model's weights and only needs the correct linguistic "Trigger" to be activated.

## 5. Advanced CoT Variants (Tree-of-Thought, etc.)
The current frontier of reasoning is the **Tree-of-Thought (ToT)** framework. Unlike linear CoT, ToT allows the model to explore multiple "Branches" of thought simultaneously. If a branch reaches a logical dead-end, the model can "Backtrack" to a previous node and try a different path. This mimics high-level human problem-solving.

## 📊 Visual Resources and Diagrams

- **Standard vs. Chain-of-Thought Comparison**: A visualization of the intermediate reasoning steps in the reasoning stream.
    - [Source: Wei et al. (2022) - Chain-of-Thought Prompting Elicits Reasoning (Fig 1)](https://arxiv.org/pdf/2201.11903.pdf)
- **The Tree-of-Thought Search Graph**: A diagram showing BFS (Breadth-First Search) over thought branches.
    - [Source: Yao et al. (2023) - Tree of Thoughts (Fig 2)](https://arxiv.org/pdf/2305.10601.pdf)

## 🐍 Technical Implementation (Python 3.14.2)

A **Self-Consistency Majority Vote** simulator for logical verification on Windows.

```python
from collections import Counter
from typing import List

def reasoning_majority_vote(model_outputs: List[str]):
    """
    Simulates the Self-Consistency logic for higher reliability.
    Compatible with Python 3.14.2.
    """
    # 1. Extract the 'Final Answer' from each reasoning path
    # (Assuming answers are at the end after 'Ans:')
    extracted_answers = [o.split("Ans:")[-1].strip() for o in model_outputs]
    
    # 2. Compute the Majority Vote
    tally = Counter(extracted_answers)
    most_common, frequency = tally.most_common(1)[0]
    
    confidence = frequency / len(model_outputs)
    
    return most_common, confidence

if __name__ == "__main__":
    # 5 independent model reasoning paths for 'What is 15 * 12?'
    mock_paths = [
        "15 * 10 is 150. 15 * 2 is 30. 150+30 is 180. Ans: 180",
        "15 times 12 is like 12 times 10 (120) and 12 times 5 (60). Ans: 180",
        "15 * 12... let's see... 150 + 40 is 190. Ans: 190", # A mistake!
        "Multiply 15 by 12. 10*15=150, 2*15=30. Total 180. Ans: 180",
        "15 * 12 is 180. Ans: 180"
    ]
    
    answer, conf = reasoning_majority_vote(mock_paths)
    print(f"Consensus Answer: {answer} (Reliability: {conf:.0%})")
```

## 📚 Postgraduate Reference Library

### Foundational Papers
- **Wei et al. (2022)**: *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*.
    - [Link to ArXiv](https://arxiv.org/abs/2201.11903)
- **Wang et al. (2022)**: *"Self-Consistency Improves Chain of Thought Reasoning in Language Models"*.
    - [Link to ArXiv](https://arxiv.org/abs/2203.11171)

### Frontier News and Updates (2025-2026)
- **OpenAI (Late 2025)**: Development of *o1-Pro*, featuring "Dynamic Trace Length"—where the model determines how many CoT steps are needed based on problem complexity.
- **NVIDIA AI Blog**: "The Compute Cost of Thought"—Analysis of why CoT increases generation cost by 5x but reduces downstream correction costs by 50x.
- **Google Research 2026**: Announcement of *ThoughtFlow-V2*, an architecture that externalizes the CoT into a symbolic logic-checker for 100% mathematical accuracy.
