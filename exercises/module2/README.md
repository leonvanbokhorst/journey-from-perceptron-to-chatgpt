# Module 2: Multi-Layer Perceptrons and Backpropagation

This module explores multi-layer perceptrons (MLPs) and the backpropagation algorithm, which together form the foundation of modern deep learning. You'll learn how to train networks with multiple layers and implement the gradient-based optimization needed for effective learning.

## Learning Objectives

- Understand the architecture of multi-layer neural networks
- Learn the backpropagation algorithm for efficient weight updates
- Implement MLPs from scratch using NumPy
- Compare different activation functions and their impact on learning
- Apply MLPs to more complex classification and regression problems
- Understand the basics of optimization in neural networks

## Exercises

This module contains the following exercises:

1. **MLP Implementation from Scratch ([`code/ex_01_mlp_backprop.py`](code/ex_01_mlp_backprop.py))**:

   - Build a multi-layer perceptron with configurable architecture
   - Implement forward and backward passes for backpropagation
   - Visualize the gradient flow through the network
   - Understand weight initialization and learning rate effects

2. **Activation Function Exploration ([`code/ex_02_activation_functions.py`](code/ex_02_activation_functions.py))**:

   - Implement and visualize different activation functions (sigmoid, tanh, ReLU, leaky ReLU)
   - Analyze the impact of activation functions on model convergence
   - Understand the vanishing gradient problem
   - Implement activation function derivatives for backpropagation

3. **Classification with MLPs ([`code/ex_03_classification.py`](code/ex_03_classification.py))**:

   - Apply MLPs to multi-class classification problems
   - Implement mini-batch gradient descent
   - Visualize decision boundaries of the trained model
   - Compare different network architectures and their performance

4. **PyTorch MLP Implementation ([`code/ex_04_pytorch_mlp.py`](code/ex_04_pytorch_mlp.py))**:
   - Implement MLPs using PyTorch for automatic differentiation
   - Compare performance and implementation effort with NumPy version
   - Use PyTorch's built-in layers and optimization modules
   - Implement early stopping and learning rate scheduling

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python code/ex_01_mlp_backprop.py
```

## Resources

- **Concept Guide**: Read [`guides/mlp_backprop_guide.md`](guides/mlp_backprop_guide.md) for an in-depth explanation of multi-layer perceptrons and the backpropagation algorithm.
- **Backpropagation Paper**: Rumelhart et al. (1986) - ["Learning representations by back-propagating errors"](https://www.nature.com/articles/323533a0)
- **Visual Explanation**: 3Blue1Brown's neural network series: [https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **Interactive Tool**: TensorFlow Playground for visualizing neural networks: [https://playground.tensorflow.org/](https://playground.tensorflow.org/)

## Previous Modules

- [Module 1: The Perceptron and Early Neural Networks](../module1/README.md)

## Next Module

- [Module 3: Recurrent Neural Networks and Sequence Modeling](../module3/README.md)
