#!/usr/bin/env python3
"""
MLP with PyTorch - Exercise 4
Implement a Multi-Layer Perceptron using PyTorch's neural network modules
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron implemented using PyTorch's nn.Module
    """

    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        """
        Initialize the MLP with specified architecture.

        Args:
            input_size: Number of input features
            hidden_sizes: List of integers representing the size of each hidden layer
            output_size: Number of output units
            activation: Activation function to use between layers
        """
        super(MLP, self).__init__()

        # Create a sequential container
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation)

        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # For binary classification, add sigmoid activation
        if output_size == 1:
            layers.append(nn.Sigmoid())

        # Sequential container for the network
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.model(x)


def create_binary_dataset():
    """
    Create a binary classification dataset (moons).
    """
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    y = y.reshape(-1, 1)  # Reshape for binary classification

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler, X, y


def compare_optimizers():
    """
    Compare different PyTorch optimizers on the classification task.
    """
    # Get dataset
    X_train, y_train, X_test, y_test, _, _, _ = create_binary_dataset()

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Optimizers to compare
    optimizers = {
        "SGD": optim.SGD,
        "SGD with momentum": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        "Adam": optim.Adam,
        "RMSprop": optim.RMSprop,
    }

    # Training parameters
    num_epochs = 100

    # Results storage
    all_losses = {}

    # Train with each optimizer
    for opt_name, opt_class in optimizers.items():
        print(f"\nTraining with {opt_name}...")

        # Create model
        model = MLP(input_size=2, hidden_sizes=[16, 16], output_size=1)

        # Create optimizer
        optimizer = opt_class(model.parameters(), lr=0.01)

        # Loss function
        criterion = nn.BCELoss()

        # Training loop
        losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss for the epoch
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save losses
        all_losses[opt_name] = losses

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            predictions = (test_outputs > 0.5).float()
            accuracy = (predictions == y_test).float().mean()

            print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Plot losses
    plt.figure(figsize=(10, 6))
    for opt_name, losses in all_losses.items():
        plt.plot(losses, label=opt_name)

    plt.title("Convergence of Different Optimizers")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return all_losses


def visualize_decision_boundary_pytorch(
    model, X, y, scaler, title="PyTorch MLP Decision Boundary"
):
    """
    Visualize decision boundary for a PyTorch model.
    """
    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Convert to tensor for PyTorch
    grid = torch.FloatTensor(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))

    # Get predictions
    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy()

    # Convert probabilities to binary predictions
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.contour(xx, yy, Z, colors="k", linewidths=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=40)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def compare_numpy_vs_pytorch():
    """
    Compare NumPy and PyTorch implementations in terms of code and performance.
    """
    # Create dataset
    X_train, y_train, X_test, y_test, scaler, X_full, y_full = create_binary_dataset()

    # Define parameters
    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 32
    hidden_size = 16

    # 1. Create PyTorch model and train
    print("\n--- Training PyTorch Model ---")
    torch_start_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )
    torch_end_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )

    if torch_start_time:
        torch_start_time.record()

    # Create model
    torch_model = MLP(input_size=2, hidden_sizes=[hidden_size], output_size=1)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate)

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    torch_losses = []

    for epoch in range(num_epochs):
        torch_model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = torch_model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        torch_losses.append(avg_loss)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    if torch_end_time:
        torch_end_time.record()
        torch.cuda.synchronize()
        torch_time = (
            torch_start_time.elapsed_time(torch_end_time) / 1000
        )  # Convert to seconds
    else:
        torch_time = None

    # Evaluate PyTorch model
    torch_model.eval()
    with torch.no_grad():
        test_outputs = torch_model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        predictions = (test_outputs > 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()

    print(f"PyTorch Model - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    if torch_time:
        print(f"PyTorch Training Time: {torch_time:.2f} seconds")

    # Visualize PyTorch decision boundary
    visualize_decision_boundary_pytorch(
        torch_model, X_full, y_full, scaler, title="PyTorch MLP Decision Boundary"
    )

    # Plot PyTorch loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(torch_losses)
    plt.title("PyTorch MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nKey differences between NumPy and PyTorch implementations:")
    print("1. Automatic differentiation: PyTorch computes gradients automatically")
    print("2. GPU acceleration: PyTorch can utilize GPUs for faster computation")
    print(
        "3. Built-in components: PyTorch provides optimizers, loss functions, and layers"
    )
    print("4. Mini-batch processing: Easy to implement with DataLoader")
    print("5. Dynamic computation graph: Easier debugging and more flexibility")


def explore_automatic_differentiation():
    """
    Demonstrate PyTorch's automatic differentiation capability.
    """
    print("\n--- Exploring Automatic Differentiation ---")

    # Create a tensor with requires_grad=True
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x = {x.item()}")

    # Define a simple computation
    y = x**3 + 3 * x**2 - 5 * x + 1
    print(f"y = x^3 + 3x^2 - 5x + 1 = {y.item()}")

    # Compute the gradient dy/dx
    y.backward()
    print(f"dy/dx at x = 2 is {x.grad.item()}")

    # Analytical derivative: dy/dx = 3x^2 + 6x - 5
    analytical_grad = 3 * (x.item() ** 2) + 6 * x.item() - 5
    print(f"Analytical result: 3x^2 + 6x - 5 = {analytical_grad}")

    print("\nThis demonstrates how PyTorch automatically computes gradients.")
    print("In a neural network, PyTorch keeps track of all operations and")
    print("computes gradients with respect to all parameters with requires_grad=True.")

    # Reset demonstration
    print("\n--- How this applies to neural networks ---")

    # Create a simple single-layer neural network
    model = nn.Linear(2, 1)

    # Print parameters before update
    w, b = model.parameters()
    w_before = w.clone().detach()
    b_before = b.clone().detach()

    print("Initial weights:")
    print(f"w = {w_before.numpy()}")
    print(f"b = {b_before.item()}")

    # Input data
    x = torch.tensor([[1.0, 2.0]])
    y_true = torch.tensor([[0.0]])

    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = nn.MSELoss()(y_pred, y_true)
    print(f"\nLoss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Print gradients
    print("\nGradients:")
    print(f"dLoss/dw = {w.grad.numpy()}")
    print(f"dLoss/db = {b.grad.item()}")

    # Apply gradients manually
    lr = 0.1
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Print updated parameters
    print("\nUpdated weights after one step:")
    print(f"w = {w.numpy()}")
    print(f"b = {b.item()}")

    # Show how much the parameters changed
    print("\nChange in weights:")
    print(f"Δw = {(w - w_before).numpy()}")
    print(f"Δb = {(b - b_before).item()}")

    print("\nThis is exactly what optimizers do automatically during training,")
    print("but with more sophisticated update rules.")


def main():
    """Main function to run all examples."""
    print("===== MLP with PyTorch Exercise =====")

    # Part 1: Compare different optimizers
    print("\n--- Part 1: Comparing Different Optimizers ---")
    compare_optimizers()

    # Part 2: Compare NumPy vs PyTorch implementations
    print("\n--- Part 2: NumPy vs PyTorch ---")
    compare_numpy_vs_pytorch()

    # Part 3: Explore automatic differentiation
    print("\n--- Part 3: Automatic Differentiation ---")
    explore_automatic_differentiation()

    print("\nThis exercise demonstrated:")
    print("1. How to build neural networks using PyTorch")
    print("2. The advantages of using deep learning frameworks")
    print("3. How automatic differentiation works")
    print("4. Different optimizers for training neural networks")


if __name__ == "__main__":
    main()
