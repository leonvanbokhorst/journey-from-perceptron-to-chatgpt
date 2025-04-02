# Module 3: Recurrent Neural Networks and Sequence Modeling

This module introduces recurrent neural networks (RNNs), which enable processing of sequential data by maintaining internal memory. You'll learn how RNNs handle variable-length input sequences and implement applications like time series prediction and text generation.

## Learning Objectives

- Understand how recurrent neural networks process sequential data
- Learn the math behind forward and backward propagation through time (BPTT)
- Implement RNN cells from scratch using NumPy
- Apply RNNs to practical sequence modeling tasks in natural language and time series
- Recognize the limitations of vanilla RNNs in capturing long-term dependencies

## Exercises

This module contains the following exercises:

1. **Basic RNN Implementation ([`code/ex_01_basic_rnn.py`](code/ex_01_basic_rnn.py))**:

   - Build a simple RNN cell from scratch using NumPy
   - Implement forward and backward propagation through time
   - Train the model to learn simple patterns
   - Visualize hidden state dynamics

2. **Time Series Prediction ([`code/ex_02_time_series_prediction.py`](code/ex_02_time_series_prediction.py))**:

   - Apply RNNs to predict future values in time series data
   - Compare different sequence lengths and their impact on prediction
   - Implement sliding window approach for time series preprocessing
   - Evaluate model performance and analyze errors

3. **Character-Level Language Model ([`code/ex_03_char_language_model.py`](code/ex_03_char_language_model.py))**:

   - Build a character-level language model using RNNs
   - Train the model on text corpus to predict next character
   - Generate new text by sampling from the trained model
   - Experiment with temperature parameter for creativity

4. **PyTorch RNN Implementation ([`code/ex_04_rnn_pytorch.py`](code/ex_04_rnn_pytorch.py))**:
   - Implement RNN models using PyTorch's built-in modules
   - Compare manual implementation with PyTorch's optimized version
   - Use DataLoader for efficient training on sequence data
   - Apply to a more complex sequence modeling task

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python code/ex_01_basic_rnn.py
```

## Resources

- **Concept Guide**: Read [`guides/rnn_guide.md`](guides/rnn_guide.md) for an in-depth explanation of RNN architecture, backpropagation through time, and applications.
- **The Unreasonable Effectiveness of RNNs**: Karpathy's blog post on character-level language models: [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- **Understanding BPTT**: Olah's visual guide to backpropagation through time: [https://colah.github.io/posts/2015-08-Backprop/](https://colah.github.io/posts/2015-08-Backprop/)
- **PyTorch Documentation**: RNN modules and functions: [https://pytorch.org/docs/stable/nn.html#recurrent-layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers)

## Previous Modules

- [Module 2: Multi-Layer Perceptrons and Backpropagation](../module2/README.md)
- [Module 1: The Perceptron and Early Neural Networks](../module1/README.md)

## Next Module

- [Module 4: Long Short-Term Memory and Gated Recurrent Units](../module4/README.md)
