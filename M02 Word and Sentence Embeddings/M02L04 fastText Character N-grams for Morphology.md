# Chapter 2.4: fastText: Character N-grams for Morphology

## 1. Handling Out-of-Vocabulary (OOV) Words
Static embedding models like Word2Vec and GloVe share a common structural weakness: each word is an "atomic unit." If a word was not present in the training set, the model generates an "Unknown" (`<|unk|>`) token, losing all semantic information. **fastText**, developed at FAIR (Facebook AI Research), introduces **Sub-lexical Representation** to resolve this. Even if a model has never seen the word "biotransformation," it can construct a highly accurate embedding by analyzing the roots and suffixes it *has* seen before.

## 2. Using Character N-grams in fastText
fastText achieves this by decomposing each word into a set of **Character N-grams**. 
- **Mechanism**: For the word "apple" with $n=3$, the model generates: `<ap`, `app`, `ppl`, `ple`, `le>`. 
- **Integration**: The final vector for "apple" is the **Centroid (Average)** of the vector for the full word and the vectors for all its constituent n-grams. This allows the model to share internal parameters between words that share the same morphological origins.

## 3. Benefits for Morphologically Rich Languages
This character-level awareness is disproportionately effective for **Morphologically Rich** languages such as Arabic, Finnish, Turkish, or German. In these languages, a single word can take hundreds of forms that may appear only once in a corpus. By learning from character-level patterns, fastText can represent these rare variations with high precision, making it the industry standard for non-English high-resolution text processing.

## 4. fastText for Text Classification
Beyond embedding generation, the fastText library is a high-performance engine for **Text Classification**. By utilizing an optimized linear model with a hierarchical softmax, it can train on billions of tokens in a matter of minutes. Its speed and inherent robustness to **Typographical Errors** (since "proccessor" shares many n-grams with "processor") make it an indispensable tool for real-time industrial applications where low-latency and reliability are paramount.
