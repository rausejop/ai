# Chapter 5.4: Reducing Information Overload: Text Summarization

## 1. Extractive vs. Abstractive Summarization
Text summarization is the process of generating a concise and coherent version of a longer document while preserving its essential semantic weight. There are two distinct technical paradigms:
- **Extractive**: Selecting and assembling the most important sentences already present in the source text. It is factually accurate but can result in disjointed narratives.
- **Abstractive**: Generating entirely new sentences that paraphrase the original meaning. This provides high-density, fluent text but carries a higher risk of hallucination.

## 2. Techniques for Extractive Summarization
Traditional extractive methods used algorithms like **TextRank** (a variation of Google's PageRank) to identify central sentences. Modern extractive systems use BERT to calculate the embedding of every sentence and select the subset that encompasses the maximum "information coverage" of the document. These are ideal for processing technical manuals or legal documents where using the original terminology is a hard requirement.

## 3. Sequence-to-Sequence (Seq2Seq) for Abstractive
Abstractive summarization is driven by **Encoder-Decoder** architectures. The encoder processes the source document into a dense "context vector," and the decoder then uses this vector as a guide to write a summary from scratch. This auto-regressive process allows the model to synthesize information from various parts of the document into a single, cohesive paragraph.

## 4. BART and T5 for Summarization
The state-of-the-art for abstractive distillation is represented by models like **BART** and **T5**. 
- **BART** (Bidirectional and Auto-Regressive Transformers) is pre-trained by corrupting text and learning to reconstruct it, making it natively proficient at "re-writing." 
- **T5** (Text-to-Text Transfer Transformer) treats every NLP task as a text-generation problem, allowing it to generalize summarization across thousands of document types with extreme flexibility.

## 5. Evaluation: The ROUGE Metric
Measuring the quality of a summary is a profound challenge. The industry standard is **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). 
- **ROUGE-1/2**: measures the overlap of words and bigrams between the machine summary and a human "gold standard."
- **ROUGE-L**: measures the "Longest Common Subsequence" to evaluate sentence flow. 
While ROUGE captures linguistic overlap, modern researchers increasingly supplement it with **LLM-as-a-Judge** scoring to evaluate the factual faithfulness and narrative quality of the condensation.
