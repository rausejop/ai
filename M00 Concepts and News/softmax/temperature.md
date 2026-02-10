# Temperature Setting in LLMs
This document explains the mathematical impact of the **Temperature** ($T$) parameter on the output of Large Language Models (LLMs).

## 1. The Transformation Process (mathematical formula)
In the final layer of a Transformer, before a model predicts a word, the model produces raw numercial scores called **logits**. To convert these values into a probability distribution that sum up to 1 (or 100%), we use a modified version of the Softmax function. When you adjust the "Temperature" parameter, the formula changes as follows:

$$\text{Probability} = \text{Softmax}\left(\frac{\text{Logit}}{T}\right)$$

In a more granular form, for each specific token $i$:
$$P(y_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

To introduce variability, in a more granular form, for each specific token $i$, Temperature is applied to the denominator of the exponent in the function:
$$P(y\_i) = \\frac{e^{z\_i / T}}{\\sum\_{j} e^{z\_j / T}}$$

Where:
* $z\_i$ represents the **logit** of a specific token.
* $T$ is the **Temperature** parameter.

## 2. The Impact of $T$ on Probability
Temperature acts as a regulator of the model's "confidence" or "sharpness":

###  Low Temperature (e.g., $T = 0.1$)

* **Mechanism:** Dividing logits by a small number **drastically increases** the gap between them before they are exponentiated.

* **Result:** Probability mass concentrates heavily on the highest-scoring token.

* **Effect:** The model becomes **deterministic** and precise. Best for technical, factual, or coding tasks.

###  High Temperature (e.g., $T = 0.8$)

* **Mechanism:** Dividing logits by a larger number **flattens the differences** between them.

* **Result:** Tokens that previously had low scores now have a significant probability of being selected.

* **Effect:** The model becomes **"creative"** and diverse. Best for storytelling, brainstorming, or roleplay.
---

## 3. Behavioral Summary
| $T$ Value | Classification | Behavior | Use Case |
| :--- | :--- | :--- | :--- |
| **0.1 - 0.3** | Conservative | Focused, repetitive, logical. | Coding, Mathematics. |
| **0.7 - 1.0** | Balanced | Fluid, natural, coherent. | General Chat, Summaries. |
| **1.2 - 1.5** | Creative | Unexpected, risky, erratic. | Poetry, Fiction. |

> **Warning:** Extreme temperatures ($T > 2.0$) usually break grammatical coherence, turning the output into a stream of disconnected words.






