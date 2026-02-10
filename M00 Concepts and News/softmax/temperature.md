# Temperature Setting in LLMs



This document explains the mathematical impact of the **Temperature** ($T$) parameter on the output of Large Language Models (LLMs).



---



## 1. The Transformation Process



Before a model predicts a word, it generates raw numerical values called **logits**. To convert these values into probabilities that sum up to 1 (or 100%), the **Softmax** function is used.



To introduce variability, Temperature is applied to the denominator of the exponent in the function:



$$P(y\_i) = \\frac{e^{z\_i / T}}{\\sum\_{j} e^{z\_j / T}}$$



Where:

* $z\_i$ represents the **logit** of a specific token.

* $T$ is the **Temperature** parameter.



---



## 2. The Impact of $T$ on Probability



Temperature acts as a regulator of the model's "confidence" or "sharpness":



### 🌡️ Low Temperature (e.g., $T = 0.1$)

* **Mechanism:** Dividing logits by a small number **drastically increases** the gap between them before they are exponentiated.

* **Result:** Probability mass concentrates heavily on the highest-scoring token.

* **Effect:** The model becomes **deterministic** and precise. Best for technical, factual, or coding tasks.



### 🌡️ High Temperature (e.g., $T = 0.8$)

* **Mechanism:** Dividing logits by a larger number **flattens the differences** between them.

* **Result:** Tokens that previously had low scores now have a significant probability of being selected.

* **Effect:** The model becomes **"creative"** and diverse. Best for storytelling, brainstorming, or roleplay.



---



## 3. Behavioral Summary







| $T$ Value | Classification | Behavior | Use Case |

| :--- | :--- | :--- | :--- |

| **0.1 - 0.3** | Conservative | Focused, repetitive, logical. | Coding, Mathematics. |

| **0.7 - 1.0** | Balanced | Fluid, natural, coherent. |



