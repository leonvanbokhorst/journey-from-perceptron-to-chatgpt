# Module 3: Recurrent Neural Networks and Sequence Modeling

This module introduces Recurrent Neural Networks (RNNs), which are designed to process sequential data by maintaining a form of memory about previous inputs. RNNs are crucial for tasks involving time series, natural language, audio processing, and other sequential data types.

## Learning Objectives

By the end of this module, you will be able to:

1. Understand the architecture and mathematical formulation of RNNs
2. Implement a basic RNN from scratch using NumPy
3. Apply RNNs to sequence prediction problems
4. Understand backpropagation through time (BPTT)
5. Create a character-level language model
6. Implement RNNs using PyTorch

## Exercises

### Exercise 1: Basic RNN Implementation

- Implement a basic RNN from scratch using NumPy
- Train it on a simple sequence prediction task
- Visualize the hidden state evolution
- Gain intuition about RNN memory and information flow

### Exercise 2: Time Series Prediction

- Apply RNNs to time series forecasting
- Prepare time series data with appropriate windowing
- Train models with different sequence lengths
- Evaluate prediction accuracy and visualize results

### Exercise 3: Character-Level Language Model

- Build a character-level language model
- Process text data for RNN training
- Train the model to predict the next character in a sequence
- Generate new text from the trained model
- Explore the impact of temperature on sampling

### Exercise 4: RNN with PyTorch

- Implement RNNs using PyTorch's built-in modules
- Compare implementation simplicity versus the from-scratch approach
- Explore PyTorch's RNN, RNNCell, and other related modules
- Apply the implementation to a real-world task

## Setup Instructions

1. Make sure you have completed the installation steps in the main README
2. Install additional dependencies if needed:
   ```
   pip install matplotlib numpy torch scikit-learn pandas
   ```
3. Navigate to this module's directory:
   ```
   cd exercises/module3
   ```

## Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Blog post by Christopher Olah
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Blog post by Andrej Karpathy
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
- Original RNN Papers:
  - Elman, J. L. (1990). Finding structure in time. Cognitive science, 14(2), 179-211.
  - Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
