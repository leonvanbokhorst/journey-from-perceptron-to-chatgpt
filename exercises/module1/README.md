# Module 1: The Perceptron and Early Neural Networks

This module covers the fundamental building block of neural networks: the perceptron. You'll learn about the perceptron model, linear separability, the perceptron learning rule, and implement your own perceptron to classify simple data.

## Learning Objectives

- Understand the basic neuron model (perceptron) including inputs, weights, bias, and binary output
- Grasp the concept of linear separability and why single-layer perceptrons can only solve linearly separable problems
- Implement the perceptron learning algorithm and train a perceptron to classify simple data
- Visualize decision boundaries and understand the limitations of perceptrons

## Exercises

This module contains the following exercises:

1. **Perceptron Basics (`code/ex_01_perceptron_basics.py`)**:

   - Implement a perceptron from scratch
   - Train it on logic gates (AND/OR)
   - Visualize the decision boundaries
   - Demonstrate the perceptron's inability to learn XOR

2. **Multi-Layer Perceptron for XOR (`code/ex_02_mlp_xor.py`)**:

   - Implement a simple 2-layer network to solve XOR (preview of Module 2)
   - Compare with the single perceptron's performance

3. **Decision Boundary Visualization (`code/ex_03_decision_boundaries.py`)**:

   - Generate and visualize different datasets
   - Train perceptrons and observe their decision boundaries
   - Explore the effect of learning rate and initialization

4. **Perceptron with PyTorch (`code/ex_04_perceptron_pytorch.py`)**:
   - Reimplement the perceptron using PyTorch
   - Compare with the NumPy implementation
   - Introduction to using a deep learning framework

## Setup

Make sure you have installed all requirements from the main project:

```bash
pip install -r ../../requirements.txt
```

## Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python exercises/module1/code/ex_01_perceptron_basics.py
```

## Resources

- **Concept Guide**: Read `guides/perceptron_guide.md` for an in-depth explanation of the perceptron, linear separability, and the learning algorithm.
- **Original Paper**: Rosenblatt (1958) – "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- **Important Critique**: Minsky & Papert (1969) – "Perceptrons"
- **Visual Explanation**: 3Blue1Brown's Neural Network series: https://www.youtube.com/watch?v=aircAruvnKk
