# Module 1: The Perceptron and Early Neural Networks

This module introduces the perceptron, the fundamental building block of neural networks. You'll learn about the historical development of artificial neurons, implement a perceptron from scratch, and understand its capabilities and limitations.

## Learning Objectives

- Understand the historical context and biological inspiration for artificial neurons
- Implement a perceptron algorithm from scratch using NumPy
- Learn how perceptrons perform binary and multi-class classification
- Visualize decision boundaries of perceptron models
- Recognize the limitations of perceptrons on non-linearly separable data
- Gain familiarity with Python tools for neural network development

## Exercises

This module contains the following exercises:

1. **Perceptron Basics ([`code/ex_01_perceptron_basics.py`](code/ex_01_perceptron_basics.py))**:

   - Implement a perceptron algorithm from scratch
   - Train the model on linearly separable datasets
   - Visualize the decision boundary and learning process
   - Experiment with different learning rates and weight initializations

2. **Multi-Class Classification ([`code/ex_02_multiclass_perceptron.py`](code/ex_02_multiclass_perceptron.py))**:

   - Extend the perceptron for multi-class classification
   - Implement one-vs-all and one-vs-one strategies
   - Compare performance on various datasets
   - Explore the convergence properties of multi-class perceptrons

3. **Perceptron Limitations ([`code/ex_03_perceptron_limitations.py`](code/ex_03_perceptron_limitations.py))**:

   - Investigate the limitations of perceptrons on XOR and other non-linearly separable problems
   - Visualize why linear models fail on these problems
   - Experiment with feature engineering to overcome basic limitations
   - Understand the need for multi-layer neural networks

4. **PyTorch Implementation ([`code/ex_04_pytorch_perceptron.py`](code/ex_04_pytorch_perceptron.py))**:
   - Implement perceptrons using PyTorch
   - Learn the basics of automatic differentiation
   - Compare implementations with NumPy and PyTorch
   - Set up a foundation for more complex neural network models

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python code/ex_01_perceptron_basics.py
```

## Resources

- **Concept Guide**: Read [`guides/perceptron_guide.md`](guides/perceptron_guide.md) for an in-depth explanation of the perceptron algorithm, its history, and mathematical foundation.
- **Original Paper**: Rosenblatt (1958) - ["The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"](https://psycnet.apa.org/record/1959-09865-001)
- **Minsky & Papert's Analysis**: Minsky & Papert (1969) - ["Perceptrons: An Introduction to Computational Geometry"](https://mitpress.mit.edu/books/perceptrons)
- **Interactive Demo**: Try an interactive perceptron visualization: [https://playground.tensorflow.org/](https://playground.tensorflow.org/)

## Next Module

- [Module 2: Multi-Layer Perceptrons and Backpropagation](../module2/README.md)
