# Module 4: Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)

## Overview

This module focuses on advanced recurrent architectures: Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU). These architectures were specifically designed to address the vanishing gradient problem in standard RNNs, allowing for effective learning of long-term dependencies in sequential data.

## Learning Objectives

By the end of this module, you will:

1. Understand the vanishing gradient problem in standard RNNs
2. Implement LSTM and GRU networks from scratch using NumPy
3. Apply LSTM/GRU networks to solve practical problems
4. Compare the performance of standard RNNs, LSTMs, and GRUs
5. Implement bidirectional and stacked LSTM/GRU architectures
6. Use PyTorch's built-in LSTM and GRU modules for efficient implementation

## Exercises

### Exercise 1: LSTM Implementation from Scratch

**File**: `code/ex_01_lstm_implementation.py`

Implement an LSTM network from scratch using NumPy. This exercise walks through each component of an LSTM cell (forget gate, input gate, cell state, output gate) and how they work together to maintain long-term memory.

### Exercise 2: GRU Implementation from Scratch

**File**: `code/ex_02_gru_implementation.py`

Implement a GRU network from scratch using NumPy. This exercise demonstrates the simplified architecture of GRUs compared to LSTMs, showing how the reset and update gates function.

### Exercise 3: Sequence Prediction with LSTM and GRU

**File**: `code/ex_03_sequence_comparison.py`

Compare the performance of standard RNNs, LSTMs, and GRUs on sequence prediction tasks of varying complexity and length. Visualize how each architecture handles long-term dependencies.

### Exercise 4: Sentiment Analysis with Bidirectional LSTM/GRU

**File**: `code/ex_04_sentiment_analysis.py`

Implement a bidirectional LSTM/GRU network for sentiment analysis on movie reviews. This exercise demonstrates the power of bidirectional architectures in NLP tasks where both past and future context matter.

## Setup Instructions

1. Make sure you have completed Module 3 on Recurrent Neural Networks
2. Ensure you have the following packages installed:
   - numpy
   - matplotlib
   - pytorch
   - pandas
   - scikit-learn

## Resources

- [Original LSTM Paper by Hochreiter & Schmidhuber](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Understanding LSTM Networks - Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [GRU Paper by Cho et al.](https://arxiv.org/abs/1406.1078)
- [PyTorch Documentation on LSTMs and GRUs](https://pytorch.org/docs/stable/nn.html#recurrent-layers)

Happy learning!
