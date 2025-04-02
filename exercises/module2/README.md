# Module 2: Multi-Layer Perceptrons and Backpropagation

This module explores how adding hidden layers to neural networks allows them to solve complex, non-linearly separable problems. You'll learn the backpropagation algorithm for training multi-layer neural networks and implement it from scratch.

## Learning Objectives

- Understand how multi-layer perceptrons overcome the limitations of single-layer networks
- Learn the backpropagation algorithm and how it efficiently computes gradients
- Implement a complete neural network with backpropagation in Python
- Explore different activation functions (sigmoid, ReLU) and their effects on training
- Visualize the learning process and decision boundaries of neural networks

## Exercises

This module contains the following exercises:

1. **MLP Implementation (`code/ex_01_mlp_implementation.py`)**:

   - Build a multi-layer perceptron from scratch using NumPy
   - Implement the forward pass with activation functions
   - Implement the backward pass (backpropagation algorithm)
   - Train the network on XOR and visualize the results

2. **Activation Functions (`code/ex_02_activation_functions.py`)**:

   - Implement different activation functions (sigmoid, tanh, ReLU)
   - Visualize these functions and their derivatives
   - Compare their performance on the same problem
   - Understand why non-linearities are crucial for deep networks

3. **Classification with MLPs (`code/ex_03_mlp_classification.py`)**:

   - Apply MLPs to more complex classification problems
   - Implement mini-batch gradient descent
   - Explore the effect of network depth and width
   - Visualize decision boundaries for different architectures

4. **MLP with PyTorch (`code/ex_04_mlp_pytorch.py`)**:
   - Implement an MLP using PyTorch's neural network modules
   - Compare with the NumPy implementation for speed and flexibility
   - Use PyTorch's automatic differentiation for backpropagation
   - Learn modern deep learning workflows

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python exercises/module2/code/ex_01_mlp_implementation.py
```

## Resources

- **Concept Guide**: Read `guides/mlp_backprop_guide.md` for an in-depth explanation of multi-layer perceptrons and the backpropagation algorithm.
- **Original Paper**: Rumelhart, Hinton & Williams (1986) – "Learning representations by back-propagating errors"
- **Online Book**: Michael Nielsen's "Neural Networks and Deep Learning" – http://neuralnetworksanddeeplearning.com/
- **Video Series**: 3Blue1Brown's videos on Gradient Descent and Backpropagation – https://www.youtube.com/watch?v=IHZwWFHWa-w
