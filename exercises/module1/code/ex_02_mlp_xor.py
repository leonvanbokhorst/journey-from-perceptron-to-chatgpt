#!/usr/bin/env python3
"""
Multi-Layer Perceptron for XOR - Exercise 2
Implement a simple 2-layer network to solve the XOR problem (preview of Module 2)
"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleNeuron:
    """
    Single neuron with a sigmoid activation function.
    """

    def __init__(self, num_inputs):
        """
        Initialize the neuron with random weights and bias.

        Args:
            num_inputs: Number of input features
        """
        # Initialize weights and bias with small random values
        self.weights = np.random.randn(num_inputs) * 0.1
        self.bias = np.random.randn() * 0.1

    def forward(self, inputs):
        """
        Compute the output of the neuron.

        Args:
            inputs: Input features (numpy array)

        Returns:
            Neuron output after applying sigmoid activation
        """
        # Calculate the weighted sum
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply sigmoid activation function
        return self.sigmoid(weighted_sum)

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x: Input value

        Returns:
            Sigmoid of x: 1/(1+e^(-x))
        """
        return 1.0 / (1.0 + np.exp(-x))


class SimpleMultiLayerPerceptron:
    """
    A basic implementation of a two-layer neural network without backpropagation.
    This version uses a manual gradient calculation and an ad-hoc learning rule.

    Note: This is a simplified version to demonstrate how multiple neurons
    can overcome the limitations of a single perceptron. A proper implementation
    with backpropagation will be covered in Module 2.
    """

    def __init__(self, num_inputs, num_hidden):
        """
        Initialize the network with layers of neurons.

        Args:
            num_inputs: Number of input features
            num_hidden: Number of neurons in the hidden layer
        """
        # Create hidden layer neurons
        self.hidden_neurons = [SimpleNeuron(num_inputs) for _ in range(num_hidden)]

        # Create output neuron (takes inputs from hidden neurons)
        self.output_neuron = SimpleNeuron(num_hidden)

        # Store dimensions for convenience
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden

        # Learning rate
        self.learning_rate = 0.1

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs: Input features (numpy array)

        Returns:
            Network output after passing through all layers
        """
        # Compute hidden layer outputs
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_neurons]
        hidden_outputs = np.array(hidden_outputs)

        # Compute output neuron's output
        final_output = self.output_neuron.forward(hidden_outputs)

        # Return both hidden layer outputs and final output
        return hidden_outputs, final_output

    def sigmoid_derivative(self, sigmoid_output):
        """
        Derivative of the sigmoid function.
        Given sigmoid(x) as input, compute sigmoid'(x).

        Args:
            sigmoid_output: Output of the sigmoid function

        Returns:
            Derivative of sigmoid at that point
        """
        return sigmoid_output * (1 - sigmoid_output)

    def train_step(self, inputs, target):
        """
        Perform a single training step (forward and backward pass).
        This implements a simple version of gradient descent.

        Args:
            inputs: Input features (numpy array)
            target: Target output (scalar)

        Returns:
            Loss for this example
        """
        # Forward pass
        hidden_outputs, final_output = self.forward(inputs)

        # Compute loss
        loss = 0.5 * (target - final_output) ** 2

        # Backward pass (manual gradient computation)
        # Output layer error
        output_error = target - final_output
        output_delta = output_error * self.sigmoid_derivative(final_output)

        # Hidden layer error
        hidden_deltas = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            # Error contribution to each hidden neuron
            error_contribution = output_delta * self.output_neuron.weights[i]
            hidden_deltas[i] = error_contribution * self.sigmoid_derivative(
                hidden_outputs[i]
            )

        # Update output neuron weights
        for i in range(self.num_hidden):
            self.output_neuron.weights[i] += (
                self.learning_rate * output_delta * hidden_outputs[i]
            )
        self.output_neuron.bias += self.learning_rate * output_delta

        # Update hidden neuron weights
        for i in range(self.num_hidden):
            for j in range(self.num_inputs):
                self.hidden_neurons[i].weights[j] += (
                    self.learning_rate * hidden_deltas[i] * inputs[j]
                )
            self.hidden_neurons[i].bias += self.learning_rate * hidden_deltas[i]

        return loss

    def train(self, X, y, epochs=10000):
        """
        Train the network on a dataset.

        Args:
            X: Training data (2D numpy array, samples x features)
            y: Target labels (1D numpy array)
            epochs: Number of training epochs

        Returns:
            History of losses
        """
        num_samples = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0

            # Train on each sample
            for i in range(num_samples):
                loss = self.train_step(X[i], y[i])
                total_loss += loss

            # Record average loss for this epoch
            avg_loss = total_loss / num_samples
            loss_history.append(avg_loss)

            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

            # Early stopping if loss is small enough
            if avg_loss < 0.001:
                print(f"Converged at epoch {epoch} with loss {avg_loss:.6f}")
                break

        return loss_history

    def predict(self, X):
        """
        Make predictions for multiple samples.

        Args:
            X: Input features (2D numpy array, samples x features)

        Returns:
            Array of predictions for each sample
        """
        predictions = []
        for inputs in X:
            _, output = self.forward(inputs)
            # Convert to binary output (0 or 1)
            binary_output = 1 if output > 0.5 else 0
            predictions.append(binary_output)

        return np.array(predictions)

    def visualize_decision_boundary(self, X, y, title="MLP Decision Boundary"):
        """
        Visualize the decision boundary for 2D data.

        Args:
            X: Input features (2D numpy array, samples x features)
            y: Target labels (1D numpy array)
            title: Plot title
        """
        # Only works for 2D inputs
        if X.shape[1] != 2:
            print("Can only visualize 2D data")
            return

        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        # Make predictions on the mesh grid
        grid_inputs = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_inputs)
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary and training points
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.contour(xx, yy, Z, colors="k", linewidths=1)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=100)

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.show()


def train_xor_mlp():
    """
    Train a multi-layer perceptron to learn the XOR function.
    """
    # Define the XOR training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR function

    # Create and train the MLP
    mlp = SimpleMultiLayerPerceptron(num_inputs=2, num_hidden=2)
    print("Training MLP for XOR function...")
    loss_history = mlp.train(X, y)

    # Visualize the learning progress
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title("MLP Learning Curve for XOR")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize the decision boundary
    mlp.visualize_decision_boundary(X, y, title="MLP for XOR Function")

    # Test the trained model
    print("\nTesting the trained MLP:")
    for inputs in X:
        _, output = mlp.forward(inputs)
        binary_output = 1 if output > 0.5 else 0
        print(
            f"{inputs[0]} XOR {inputs[1]} = {binary_output} (raw output: {output:.4f})"
        )


def main():
    """Main function to run the exercise."""
    print("===== Multi-Layer Perceptron (MLP) for XOR Exercise =====")
    print("\nThis demonstrates how a simple 2-layer neural network")
    print("can solve the XOR problem that a single perceptron cannot.")
    print("\nThe network consists of:")
    print("- 2 input neurons (for the 2 input features)")
    print("- 2 hidden neurons (with sigmoid activation)")
    print("- 1 output neuron (with sigmoid activation)")

    train_xor_mlp()

    print("\nUnlike a single perceptron, the MLP can create a non-linear")
    print("decision boundary that separates the XOR data points correctly.")
    print("\nThis is a preview of what we'll learn in Module 2, where we'll")
    print("introduce the formal backpropagation algorithm for training MLPs.")


if __name__ == "__main__":
    main()
