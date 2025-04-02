# Module 4: Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)

This module explores advanced recurrent neural network architectures that address the vanishing gradient problem in standard RNNs. You'll learn how LSTMs and GRUs use gating mechanisms to control information flow, enabling models to capture long-term dependencies in sequence data.

## Learning Objectives

- Understand the vanishing gradient problem that limits standard RNNs
- Learn how LSTM cells use forget, input, and output gates to control information flow
- Implement GRU architecture and understand its simplified gating mechanism
- Compare the performance of different RNN variants on sequence tasks
- Apply bidirectional RNNs to utilize both past and future context

## Exercises

This module contains the following exercises:

1. **LSTM Implementation ([`code/ex_01_lstm_implementation.py`](code/ex_01_lstm_implementation.py))**:

   - Build an LSTM cell from scratch using NumPy
   - Understand the purpose of each gate and how they interact
   - Implement forward and backward passes
   - Visualize gate activations to understand LSTM behavior

2. **GRU Implementation ([`code/ex_02_gru_implementation.py`](code/ex_02_gru_implementation.py))**:

   - Implement a GRU cell from scratch
   - Understand the reset and update gates
   - Compare the computational efficiency with LSTM
   - Visualize internal gate dynamics during processing

3. **RNN Architecture Comparison ([`code/ex_03_architecture_comparison.py`](code/ex_03_architecture_comparison.py))**:

   - Evaluate vanilla RNN, LSTM, and GRU on sequence prediction tasks
   - Compare performance on short vs. long sequences
   - Analyze training speed, convergence, and memory usage
   - Visualize hidden state dynamics across architectures

4. **Sentiment Analysis with Bidirectional RNNs ([`code/ex_04_bidirectional_sentiment.py`](code/ex_04_bidirectional_sentiment.py))**:
   - Implement bidirectional LSTM and GRU models using PyTorch
   - Apply to sentiment analysis on a text dataset
   - Explore effects of hyperparameters on model performance
   - Analyze how bidirectionality improves context understanding

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python code/ex_01_lstm_implementation.py
```

## Resources

- **Concept Guide**: Read [`guides/lstm_gru_guide.md`](guides/lstm_gru_guide.md) for an in-depth explanation of LSTM and GRU architectures, their advantages, and use cases.
- **LSTM Paper**: Hochreiter & Schmidhuber (1997) - ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **GRU Paper**: Cho et al. (2014) - ["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078)
- **Bidirectional RNNs**: Schuster & Paliwal (1997) - ["Bidirectional Recurrent Neural Networks"](https://ieeexplore.ieee.org/document/650093)

## Previous Modules

- [Module 3: Recurrent Neural Networks and Sequence Modeling](../module3/README.md)
- [Module 2: Multi-Layer Perceptrons and Backpropagation](../module2/README.md)

## Next Module

- [Module 5: Sequence-to-Sequence Learning and Attention Mechanisms](../module5/README.md)
