#!/usr/bin/env python3
"""
Perceptron Basics - Exercise 1
Implement a perceptron from scratch and train it on logic gates (AND/OR)
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Implementation of a Perceptron classifier.

    A perceptron is a simple binary classifier that applies weights to inputs,
    adds a bias, and produces a binary output based on a threshold.
    """

    def __init__(self, num_features, learning_rate=0.1):
        """
        Initialize the perceptron with random weights and bias.

        Args:
            num_features: Number of input features
            learning_rate: How quickly the model learns (default 0.1)
        """
        # Initialize weights with small random values
        self.weights = np.random.randn(num_features) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate

    def predict_single(self, inputs):
        """
        Make a prediction for a single sample.

        Args:
            inputs: Input features (numpy array)

        Returns:
            1 if weighted sum + bias > 0, else 0
        """
        # Calculate the weighted sum of inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply step function
        return 1 if weighted_sum > 0 else 0

    def predict(self, X):
        """
        Make predictions for multiple samples.

        Args:
            X: Input features (2D numpy array, samples x features)

        Returns:
            Array of predictions (1 or 0 for each sample)
        """
        return np.array([self.predict_single(x) for x in X])

    def train(self, X, y, max_epochs=100):
        """
        Train the perceptron using the perceptron learning algorithm.

        Args:
            X: Training data (2D numpy array, samples x features)
            y: Target labels (1 or 0 for each sample)
            max_epochs: Maximum number of training epochs

        Returns:
            Number of epochs it took to converge, and history of errors
        """
        num_samples = X.shape[0]
        error_history = []

        for epoch in range(max_epochs):
            total_error = 0

            # Iterate through each training sample
            for i in range(num_samples):
                # Make a prediction
                prediction = self.predict_single(X[i])

                # Calculate error
                error = y[i] - prediction
                total_error += abs(error)

                # Update weights and bias only if prediction is wrong
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error

            # Record the total error for this epoch
            error_history.append(total_error)

            # If no errors were made, the perceptron has converged
            if total_error == 0:
                print(f"Converged after {epoch+1} epochs!")
                return epoch + 1, error_history

        print(f"Did not converge within {max_epochs} epochs.")
        return max_epochs, error_history

    def visualize_decision_boundary(self, X, y, title="Perceptron Decision Boundary"):
        """
        Visualize the decision boundary for 2D data.

        Args:
            X: Input features (2D numpy array with 2 features)
            y: Target labels
            title: Plot title
        """
        # Only works for 2D data
        if X.shape[1] != 2:
            print("Can only visualize 2D data")
            return

        # Create a mesh grid to visualize the decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        # Make predictions on the mesh grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary and training points
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.contour(xx, yy, Z, colors="k", linewidths=1)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=100)

        # Add equation of the decision boundary to the plot
        w1, w2 = self.weights
        b = self.bias
        plt.title(f"{title}\nDecision boundary: {w1:.2f}*x + {w2:.2f}*y + {b:.2f} = 0")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.show()


def train_logic_gate(gate="AND"):
    """
    Train a perceptron to learn the AND or OR logic gate.

    Args:
        gate: The logic gate to learn ('AND' or 'OR')
    """
    # Define the training data for the logic gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Define target labels based on the selected gate
    if gate == "AND":
        y = np.array([0, 0, 0, 1])  # AND gate
    elif gate == "OR":
        y = np.array([0, 1, 1, 1])  # OR gate
    else:
        raise ValueError("Unsupported gate. Use 'AND' or 'OR'.")

    # Initialize and train the perceptron
    perceptron = Perceptron(num_features=2)
    print(f"Training perceptron for {gate} gate...")
    epochs, error_history = perceptron.train(X, y)

    # Print the final weights and bias
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")

    # Visualize the learning progress
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(error_history) + 1), error_history, marker="o")
    plt.title(f"Learning curve for {gate} gate")
    plt.xlabel("Epoch")
    plt.ylabel("Number of misclassifications")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize the decision boundary
    perceptron.visualize_decision_boundary(X, y, title=f"Perceptron for {gate} gate")

    # Test the trained perceptron
    print("\nTesting the trained perceptron:")
    for inputs in X:
        prediction = perceptron.predict_single(inputs)
        print(f"{inputs[0]} {gate} {inputs[1]} = {prediction}")


def try_xor_gate():
    """
    Demonstrate the perceptron's inability to learn the XOR function.
    """
    # Define the training data for the XOR gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR gate

    # Initialize and train the perceptron
    perceptron = Perceptron(num_features=2)
    print("Training perceptron for XOR gate...")
    epochs, error_history = perceptron.train(X, y, max_epochs=100)

    # Print the final weights and bias
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")

    # Visualize the learning progress
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(error_history) + 1), error_history, marker="o")
    plt.title("Learning curve for XOR gate")
    plt.xlabel("Epoch")
    plt.ylabel("Number of misclassifications")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize the decision boundary
    perceptron.visualize_decision_boundary(X, y, title="Perceptron attempt at XOR gate")

    # Test the trained perceptron
    print("\nTesting the trained perceptron:")
    correct = 0
    for i, inputs in enumerate(X):
        prediction = perceptron.predict_single(inputs)
        print(f"{inputs[0]} XOR {inputs[1]} = {prediction} (should be {y[i]})")
        if prediction == y[i]:
            correct += 1

    print(f"\nAccuracy: {correct/len(y)*100:.1f}%")
    print("\nAs expected, a single perceptron cannot learn the XOR function!")
    print("This is because XOR is not linearly separable.")
    print(
        "In the next exercise, we'll see how a multi-layer perceptron can solve this problem."
    )


def main():
    """Main function to run the exercise."""
    print("===== Perceptron Basics Exercise =====")
    print("\n1. Training perceptron for AND gate")
    train_logic_gate("AND")

    print("\n2. Training perceptron for OR gate")
    train_logic_gate("OR")

    print("\n3. Attempting to train perceptron for XOR gate")
    try_xor_gate()


if __name__ == "__main__":
    main()
