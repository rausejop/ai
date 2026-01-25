# Chapter 3.4: Deep Dive: BERT and Encoder-Only Models

## 1. Bidirectionality and its Power
**BERT** (Bidirectional Encoder Representations from Transformers) achieved state-of-the-art success by fully embracing the power of **Deep Bidirectionality**. While earlier models processed text linearly (left-to-right), BERT analyzes the relationship of a token to all its neighbors simultaneously across all layers of the encoder stack. This allows it to capture the subtle nuances of polysemy and complex syntactic structures that are inherently lost in unidirectional processing.

## 2. BERT's Pre-training Tasks (MLM and NSP)
BERT's intelligence is forged through two self-supervised objectives:
- **Masked Language Modeling (MLM)**: 15% of tokens are replaced with a `[MASK]` token. The model must predict these hidden units using the surrounding context. This forces it to learn an "internal dictionary" of the language.
- **Next Sentence Prediction (NSP)**: The model is shown pairs of sentences and must determine if Sentence B follows Sentence A. This captures document-level coherence and relationship.

## 3. RoBERTa: Optimized Pre-training of BERT
**RoBERTa** (Robustly Optimized BERT Pretraining Approach) proved that BERT was significantly "under-trained." By removing the NSP task (which was found to be less effective), using 10x more data, and significantly larger batch sizes, RoBERTa achieved substantial performance gains. A key technical innovation in RoBERTa is **Dynamic Masking**, where the masked tokens change in every iteration, preventing the model from memorizing specific patterns and forcing a more robust semantic representation.

## 4. Use Cases: Classification and Named Entity Recognition
Encoder-only models remain the gold standard for high-precision, non-generative tasks.
- **Classification**: By prepending the `[CLS]` token, BERT creates a high-density "summary vector" of any sequence, which is then fine-tuned to classify sentiment, intent, or topic.
- **Named Entity Recognition (NER)**: BERT's token-level representations allow for precise labeling of entities (People, Places, Organizations). Because it understands the full sentence, it can resolve ambiguity (distinguishing "Apple" the company from "apple" the fruit) with near-human accuracy.
