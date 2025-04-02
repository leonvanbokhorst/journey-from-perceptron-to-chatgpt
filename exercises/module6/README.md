# Module 6: The Transformer Architecture – "Attention Is All You Need"

This module explores the revolutionary Transformer architecture as proposed in the "Attention Is All You Need" paper. You'll learn how Transformers use self-attention mechanisms to process sequences in parallel, eliminating the need for recurrence and convolutions while achieving superior performance.

## Learning Objectives

- Understand how Transformers represent a fundamental shift by replacing recurrence with self-attention
- Learn the components of the Transformer architecture: multi-head self-attention, positional encoding, encoder and decoder blocks
- Comprehend why Transformers can process sequences in parallel and how they handle long-range dependencies
- Implement a simplified Transformer model and experiment with it on sequence tasks
- Recognize the significance of Transformers as the foundation for modern large language models

## Exercises

This module contains the following exercises:

1. **Self-Attention Mechanism (`code/ex_01_self_attention.py`)**:

   - Implement the scaled dot-product attention mechanism
   - Visualize attention patterns between sequence elements
   - Experiment with different attention patterns and masking

2. **Multi-Head Attention (`code/ex_02_multi_head_attention.py`)**:

   - Build a multi-head attention module
   - Understand how multiple attention heads capture different relationships
   - Analyze and visualize what different heads learn

3. **Transformer Encoder (`code/ex_03_transformer_encoder.py`)**:

   - Implement a full Transformer encoder block
   - Build a stack of encoder layers
   - Process sequences and visualize the learned representations

4. **Complete Transformer (`code/ex_04_complete_transformer.py`)**:
   - Implement a simplified Transformer architecture with encoder and decoder
   - Apply the model to a sequence-to-sequence task
   - Analyze the performance and attention patterns in the model

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python exercises/module6/code/ex_01_self_attention.py
```

## Resources

- **Concept Guide**: Read `guides/transformer_guide.md` for an in-depth explanation of the Transformer architecture, self-attention mechanisms, and positional encodings.
- **Original Paper**: Vaswani et al. (2017) - "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
- **Visual Explanation**: The Illustrated Transformer by Jay Alammar: https://jalammar.github.io/illustrated-transformer/
- **Code Walkthrough**: The Annotated Transformer by Harvard NLP: http://nlp.seas.harvard.edu/annotated-transformer/
