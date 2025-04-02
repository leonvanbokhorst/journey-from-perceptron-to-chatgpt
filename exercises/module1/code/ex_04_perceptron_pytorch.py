#!/usr/bin/env python3
"""
Perceptron with PyTorch - Exercise 4
Reimplement the perceptron using PyTorch and compare with the NumPy implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class PyTorchPerceptron(nn.Module):
    """
    Implementation of a Perceptron classifier using PyTorch.
    """

    def __init__(self, num_features):
        """
        Initialize the perceptron with random weights and bias.

        Args:
            num_features: Number of input features
        """
        super(PyTorchPerceptron, self).__init__()

        # Create a single linear layer (no hidden layers, no activation function yet)
        self.linear = nn.Linear(num_features, 1)

        # Initialize weights with small random values, similar to our NumPy implementation
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)

        # Store a history of errors for visualization
        self.error_history = []

    def forward(self, x):
        """
        Forward pass through the perceptron.

        Args:
            x: Input tensor of shape (batch_size, num_features)

        Returns:
            Raw output from the linear layer
        """
        return self.linear(x)

    def predict(self, x):
        """
        Make predictions using the perceptron (apply threshold to get binary output).

        Args:
            x: Input tensor or NumPy array

        Returns:
            Binary predictions (0 or 1)
        """
        # Convert input to tensor if it's a NumPy array
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)

        # Forward pass
        with torch.no_grad():
            output = self.forward(x)
            # Apply step function (threshold at 0)
            predictions = (output > 0).float()

        # Return as NumPy array for consistency with the other implementation
        return predictions.numpy().flatten()

    def train_model(self, X, y, learning_rate=0.1, max_epochs=100):
        """
        Train the perceptron using a manual implementation of the perceptron learning rule.
        This mimics the behavior of our NumPy implementation.

        Args:
            X: Training data (NumPy array)
            y: Target labels (NumPy array)
            learning_rate: Learning rate for weight updates
            max_epochs: Maximum number of training epochs

        Returns:
            Number of epochs it took to converge, and error history
        """
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))

        # Reset error history
        self.error_history = []

        for epoch in range(max_epochs):
            total_error = 0

            # Process each sample individually to mimic the perceptron algorithm
            for i in range(len(X)):
                x_i = X_tensor[i : i + 1]  # Get a single sample (keep batch dimension)
                y_i = y_tensor[i]  # Get corresponding target

                # Forward pass
                output = self.forward(x_i)

                # Apply step function to get binary prediction
                prediction = (output > 0).float()

                # Compute error
                error = y_i - prediction

                # Update weights if prediction is wrong (manual weight update)
                if error != 0:
                    # Manually update weights using the perceptron learning rule
                    with torch.no_grad():
                        self.linear.weight += learning_rate * error * x_i
                        self.linear.bias += learning_rate * error

                total_error += abs(error.item())

            # Record the total error for this epoch
            self.error_history.append(total_error)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Error: {total_error}")

            # If no errors were made, the perceptron has converged
            if total_error == 0:
                print(f"Converged after {epoch+1} epochs!")
                return epoch + 1, self.error_history

        print(f"Did not converge within {max_epochs} epochs.")
        return max_epochs, self.error_history

    def train_using_pytorch_optimizer(self, X, y, learning_rate=0.1, max_epochs=1000):
        """
        Train the perceptron using PyTorch's built-in optimization methods.
        This demonstrates how to use PyTorch's automatic differentiation and optimizers.

        Args:
            X: Training data (NumPy array)
            y: Target labels (NumPy array)
            learning_rate: Learning rate for the optimizer
            max_epochs: Maximum number of training epochs

        Returns:
            Loss history during training
        """
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # For tracking progress
        loss_history = []

        for epoch in range(max_epochs):
            # Forward pass
            outputs = self.forward(X_tensor)

            # Compute loss
            loss = criterion(outputs, y_tensor)
            loss_history.append(loss.item())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Check if the model has converged (all predictions correct)
            with torch.no_grad():
                predictions = (outputs > 0).float()
                accuracy = (predictions == y_tensor).float().mean().item()
                if accuracy == 1.0:
                    print(f"Converged after {epoch+1} epochs!")
                    break

        return loss_history

    def visualize_decision_boundary(
        self, X, y, title="PyTorch Perceptron Decision Boundary"
    ):
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

        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        # Make predictions on the mesh grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary and training points
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.contour(xx, yy, Z, colors="k", linewidths=1)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=100)

        # Extract weights and bias for the boundary equation
        weight = self.linear.weight.detach().numpy().flatten()
        bias = self.linear.bias.item()

        plt.title(
            f"{title}\nDecision boundary: {weight[0]:.2f}*x + {weight[1]:.2f}*y + {bias:.2f} = 0"
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.show()


def train_logic_gate_pytorch(gate="AND", use_optimizer=False):
    """
    Train a PyTorch perceptron to learn the AND or OR logic gate.

    Args:
        gate: The logic gate to learn ('AND' or 'OR')
        use_optimizer: Whether to use PyTorch's optimizer or manual perceptron learning
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

    # Initialize the PyTorch perceptron
    perceptron = PyTorchPerceptron(num_features=2)
    print(f"Training PyTorch perceptron for {gate} gate...")

    # Train the model
    if use_optimizer:
        print("Using PyTorch optimizer (SGD with BCEWithLogitsLoss)")
        loss_history = perceptron.train_using_pytorch_optimizer(X, y)

        # Plot the loss curve
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.title(f"Loss curve for {gate} gate (PyTorch optimizer)")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Using manual perceptron learning rule")
        epochs, error_history = perceptron.train_model(X, y)

        # Plot the error curve
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(error_history) + 1), error_history, marker="o")
        plt.title(f"Learning curve for {gate} gate (Perceptron rule)")
        plt.xlabel("Epoch")
        plt.ylabel("Number of misclassifications")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Print the learned weights and bias
    print(f"Learned weights: {perceptron.linear.weight.detach().numpy().flatten()}")
    print(f"Learned bias: {perceptron.linear.bias.item()}")

    # Visualize the decision boundary
    perceptron.visualize_decision_boundary(
        X, y, title=f"PyTorch Perceptron for {gate} gate"
    )

    # Test the trained perceptron
    print("\nTesting the trained perceptron:")
    for inputs in X:
        prediction = perceptron.predict(inputs.reshape(1, -1))[0]
        print(f"{inputs[0]} {gate} {inputs[1]} = {prediction}")


def compare_with_numpy_implementation():
    """
    Compare the PyTorch implementation with the NumPy implementation.
    """
    print("This exercise implements a perceptron using PyTorch.")
    print("The key differences between the PyTorch and NumPy implementations are:")
    print("\n1. PyTorch provides automatic differentiation (autograd)")
    print("   - We can use optimizers like SGD instead of manual weight updates")
    print("   - We can use loss functions like BCEWithLogitsLoss")
    print("\n2. PyTorch models can easily be moved to GPU for faster training")
    print("\n3. PyTorch is part of a larger ecosystem of deep learning tools")
    print("   - It's designed to scale to much more complex models")
    print("   - It provides layers, activations, optimizers, etc. out of the box")
    print("\nWhile a simple perceptron doesn't showcase many of PyTorch's")
    print("advantages, this introduction will be helpful as we move to more")
    print("complex models in the coming modules.")


def main():
    """Main function to run the exercise."""
    print("===== Perceptron with PyTorch Exercise =====")

    # Compare the implementations conceptually
    compare_with_numpy_implementation()

    # Train AND gate using manual perceptron rule
    print("\n1. Training perceptron for AND gate (manual perceptron learning)")
    train_logic_gate_pytorch("AND", use_optimizer=False)

    # Train OR gate using PyTorch optimizer
    print("\n2. Training perceptron for OR gate (PyTorch optimizer)")
    train_logic_gate_pytorch("OR", use_optimizer=True)

    print("\nNotice that both approaches converge to similar decision boundaries.")
    print("However, the PyTorch optimizer approach is more flexible and can be")
    print("extended to more complex models, as we'll see in later modules.")


if __name__ == "__main__":
    main()
