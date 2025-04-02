# Module 5: Sequence-to-Sequence Learning and Attention Mechanisms

This module explores sequence-to-sequence learning and attention mechanisms, which were key developments toward modern transformer-based architectures. You'll learn how the addition of attention allows models to focus on relevant parts of the input sequence, significantly improving performance on tasks like machine translation.

## Learning Objectives

- Understand the limitations of vanilla sequence-to-sequence models with encoder-decoder architecture
- Learn how attention mechanisms allow models to selectively focus on parts of the input sequence
- Implement attention mechanisms to improve sequence-to-sequence model performance
- Visualize attention patterns to gain insights into how models process sequence data
- Apply attention-based models to tasks like machine translation and text summarization

## Exercises

This module contains the following exercises:

1. **Basic Sequence-to-Sequence Model ([`code/ex_01_seq2seq_basics.py`](code/ex_01_seq2seq_basics.py))**:

   - Implement a vanilla encoder-decoder architecture from scratch
   - Train the model on a simple sequence translation task
   - Observe the limitations when dealing with long sequences
   - Compare different encoder and decoder architectures

2. **Attention Mechanism ([`code/ex_02_attention_mechanism.py`](code/ex_02_attention_mechanism.py))**:

   - Add attention mechanisms to the sequence-to-sequence model
   - Implement different attention types (Bahdanau/Additive vs. Luong/Multiplicative)
   - Visualize attention weights to see which input tokens the model focuses on
   - Compare performance against the basic sequence-to-sequence model

3. **Transformer Components ([`code/ex_03_transformer_components.py`](code/ex_03_transformer_components.py))**:

   - Implement fundamental components of the Transformer architecture
   - Explore self-attention mechanisms (foundation for Transformers)
   - Build multi-head attention modules
   - Understand positional encoding and feed-forward networks

4. **Neural Machine Translation ([`code/ex_04_neural_machine_translation.py`](code/ex_04_neural_machine_translation.py))**:
   - Apply sequence-to-sequence with attention to a translation task
   - Process and prepare parallel text datasets
   - Implement beam search for better generation
   - Evaluate translation quality using BLEU score

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python code/ex_01_seq2seq_basics.py
```

## Resources

- **Concept Guide**: Read [`guides/seq2seq_attention_guide.md`](guides/seq2seq_attention_guide.md) for an in-depth explanation of sequence-to-sequence learning and attention mechanisms.
- **Sequence-to-Sequence Learning**: Sutskever et al. (2014) - ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215)
- **Attention Mechanisms**: Bahdanau et al. (2015) - ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473)
- **Attention Variants**: Luong et al. (2015) - ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025)

## Previous Modules

- [Module 4: Long Short-Term Memory and Gated Recurrent Units](../module4/README.md)
- [Module 3: Recurrent Neural Networks and Sequence Modeling](../module3/README.md)

## Next Module

- [Module 6: The Transformer Architecture](../module6/README.md)
