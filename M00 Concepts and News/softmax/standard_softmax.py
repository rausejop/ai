import numpy as np

def softmax(logits, temperature=1.0):
    """
    Compute softmax values for each sets of scores in logits.
    
    Parameters:
    logits: list or np.array of raw scores
    temperature: float, controls the smoothness of the probability distribution
    """
    # Apply temperature scaling
    # We divide by temperature before exponentiating
    e_x = np.exp((logits - np.max(logits)) / temperature)
    
    return e_x / e_x.sum(axis=0)

# Example: LLM logits for 4 possible next words
# ["Apple", "Banana", "Cat", "Dog"]
example_logits = np.array([2.0, 1.0, 0.1, 0.5])

print(f"Probabilities (T=1.0): {softmax(example_logits, temperature=1.0)}")
print(f"Probabilities (T=0.5): {softmax(example_logits, temperature=0.5)} (More focused)")
print(f"Probabilities (T=2.0): {softmax(example_logits, temperature=2.0)} (More diverse)")