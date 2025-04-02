#!/usr/bin/env python3
"""
Activation Functions - Exercise 2
Implement and visualize different activation functions and compare their performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time


class ActivationFunction:
    """
    Base class for activation functions.
    """

    def __init__(self, name):
        self.name = name

    def forward(self, x):
        """Compute the activation value."""
        raise NotImplementedError

    def derivative(self, x):
        """Compute the derivative of the activation function."""
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    """

    def __init__(self):
        super().__init__("Sigmoid")

    def forward(self, x):
        """
        Forward pass of sigmoid.

        Args:
            x: Input array

        Returns:
            1 / (1 + e^(-x))
        """
        # Clip x to avoid overflow in exp
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def derivative(self, x):
        """
        Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))

        Args:
            x: Input array

        Returns:
            Derivative at x
        """
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function: f(x) = tanh(x)
    """

    def __init__(self):
        super().__init__("Tanh")

    def forward(self, x):
        """
        Forward pass of tanh.

        Args:
            x: Input array

        Returns:
            tanh(x)
        """
        return np.tanh(x)

    def derivative(self, x):
        """
        Derivative of tanh: f'(x) = 1 - tanh^2(x)

        Args:
            x: Input array

        Returns:
            Derivative at x
        """
        tanh_x = self.forward(x)
        return 1 - tanh_x**2


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit activation function: f(x) = max(0, x)
    """

    def __init__(self):
        super().__init__("ReLU")

    def forward(self, x):
        """
        Forward pass of ReLU.

        Args:
            x: Input array

        Returns:
            max(0, x)
        """
        return np.maximum(0, x)

    def derivative(self, x):
        """
        Derivative of ReLU: f'(x) = 1 if x > 0 else 0

        Args:
            x: Input array

        Returns:
            Derivative at x
        """
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU activation function: f(x) = max(alpha*x, x)
    """

    def __init__(self, alpha=0.01):
        super().__init__("Leaky ReLU")
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass of Leaky ReLU.

        Args:
            x: Input array

        Returns:
            max(alpha*x, x)
        """
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        """
        Derivative of Leaky ReLU: f'(x) = 1 if x > 0 else alpha

        Args:
            x: Input array

        Returns:
            Derivative at x
        """
        return np.where(x > 0, 1, self.alpha)


def visualize_activations():
    """
    Visualize different activation functions and their derivatives.
    """
    # Create a list of activation functions to visualize
    activations = [Sigmoid(), Tanh(), ReLU(), LeakyReLU(alpha=0.1)]

    # Input range for visualization
    x = np.linspace(-5, 5, 1000)

    # Create a figure with two subplots (activation and derivative)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot each activation function
    for activation in activations:
        y = activation.forward(x)
        dy = activation.derivative(x)

        ax1.plot(x, y, label=activation.name)
        ax2.plot(x, dy, label=f"{activation.name} derivative")

    # Configure plots
    ax1.set_title("Activation Functions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.grid(True)
    ax1.legend()

    ax2.set_title("Derivatives")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f'(x)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


class NeuralNetwork:
    """
    Simple neural network implementation with configurable activation function.
    """

    def __init__(self, input_size, hidden_size, output_size, activation):
        """
        Initialize neural network with one hidden layer.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            activation: Activation function for hidden layer
        """
        # Network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        # For storing intermediate values
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None  # Output

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input data (batch_size, input_size)

        Returns:
            Output predictions
        """
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation.forward(self.z1)

        # Output layer (using sigmoid for binary classification)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid for output

        return self.a2

    def backward(self, X, y, learning_rate=0.01):
        """
        Backward pass (compute gradients and update weights).

        Args:
            X: Input data
            y: Target values
            learning_rate: Learning rate for updates
        """
        m = X.shape[0]  # Batch size

        # Output layer error
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer error
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation.derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """
        Train the neural network.

        Args:
            X: Training data
            y: Target values
            epochs: Number of training iterations
            learning_rate: Learning rate for gradient descent
            verbose: Whether to print progress

        Returns:
            History of loss values
        """
        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(
                y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)
            )
            loss_history.append(loss)

            # Backward pass
            self.backward(X, y, learning_rate)

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return loss_history


def compare_activations_on_binary_classification():
    """
    Compare different activation functions on a binary classification task.
    """
    # Generate a synthetic dataset
    np.random.seed(42)
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    y = y.reshape(-1, 1)  # Reshape to (n_samples, 1)

    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Split data into train and test sets
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

    # Create a list of activation functions to compare
    activations = [Sigmoid(), Tanh(), ReLU(), LeakyReLU(alpha=0.1)]

    # Train and evaluate a network with each activation function
    plt.figure(figsize=(15, 10))
    results = []

    for i, activation in enumerate(activations):
        print(f"\nTraining with {activation.name} activation:")

        # Create and train the network
        start_time = time.time()
        network = NeuralNetwork(
            input_size=2, hidden_size=16, output_size=1, activation=activation
        )
        loss_history = network.train(
            X_train, y_train, epochs=1000, learning_rate=0.05, verbose=False
        )
        training_time = time.time() - start_time

        # Make predictions on test set
        y_pred = network.forward(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred_binary == y_test) * 100

        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"Training time: {training_time:.4f} seconds")

        # Store results
        results.append(
            {
                "name": activation.name,
                "loss_history": loss_history,
                "accuracy": accuracy,
                "training_time": training_time,
            }
        )

        # Plot the learning curve
        plt.subplot(2, 2, i + 1)
        plt.plot(loss_history)
        plt.title(f"{activation.name} - Accuracy: {accuracy:.2f}%")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot activation comparison
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(
            result["loss_history"],
            label=f"{result['name']} (Acc: {result['accuracy']:.1f}%)",
        )

    plt.title("Activation Functions Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    print(
        f"{'Activation':<15} {'Final Loss':<15} {'Accuracy':<15} {'Training Time':<15}"
    )
    print("-" * 60)
    for result in results:
        print(
            f"{result['name']:<15} {result['loss_history'][-1]:<15.6f} {result['accuracy']:<15.2f} {result['training_time']:<15.4f}"
        )


def visualize_gradient_flow():
    """
    Visualize how gradients flow through different activation functions.
    This helps demonstrate issues like the vanishing gradient problem.
    """
    # Create a deep network path (multiple layers in sequence)
    depth = 50  # Number of 'layers'

    # Create activation functions
    sigmoid = Sigmoid()
    tanh = Tanh()
    relu = ReLU()

    # Input range
    x = np.linspace(-5, 5, 1000)

    # Initialize gradients for each activation
    sigmoid_grad = np.ones_like(x)
    tanh_grad = np.ones_like(x)
    relu_grad = np.ones_like(x)

    # Compute gradient flow through layers
    for _ in range(depth):
        # Apply chain rule multiple times (simulate backpropagation through many layers)
        sigmoid_grad *= sigmoid.derivative(x)
        tanh_grad *= tanh.derivative(x)
        relu_grad *= relu.derivative(x)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid_grad, label="Sigmoid gradient after 50 layers")
    plt.plot(x, tanh_grad, label="Tanh gradient after 50 layers")
    plt.plot(x, relu_grad, label="ReLU gradient after 50 layers")

    plt.title("Gradient Magnitude After 50 Layers")
    plt.xlabel("Initial Input Value")
    plt.ylabel("Gradient Magnitude")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Log scale to better visualize vanishing gradients
    plt.tight_layout()
    plt.show()

    print("\nObservations about gradient flow:")
    print(
        "1. Sigmoid gradients vanish quickly for inputs far from 0 (saturated neurons)."
    )
    print("2. Tanh handles saturation better than sigmoid but still vanishes.")
    print("3. ReLU maintains gradient for positive inputs (no vanishing).")
    print("4. ReLU has 'dying neuron' problem - zero gradient for negative inputs.")


def main():
    """Main function to run the exercise."""
    print("===== Activation Functions Exercise =====")

    print("\n1. Visualizing Activation Functions and Their Derivatives")
    visualize_activations()

    print("\n2. Comparing Activation Functions on a Classification Task")
    compare_activations_on_binary_classification()

    print("\n3. Visualizing Gradient Flow Through Activation Functions")
    visualize_gradient_flow()

    print("\nKey Takeaways:")
    print(
        "1. Sigmoid and tanh were historically popular but can cause vanishing gradients"
    )
    print("2. ReLU is computationally efficient and helps with deep networks")
    print("3. Leaky ReLU addresses the 'dying neuron' problem of standard ReLU")
    print(
        "4. The choice of activation function affects training speed and model performance"
    )


if __name__ == "__main__":
    main()
