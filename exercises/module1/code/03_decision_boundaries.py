#!/usr/bin/env python3
"""
Decision Boundary Visualization - Exercise 3
Generate and visualize different datasets and perceptron decision boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split


class Perceptron:
    """
    Implementation of the Perceptron classifier from Exercise 1 (abridged version).
    """

    def __init__(self, num_features, learning_rate=0.1):
        # Initialize weights and bias
        self.weights = np.random.randn(num_features) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate

    def predict_single(self, inputs):
        # Calculate weighted sum and apply step function
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def train(self, X, y, max_epochs=100):
        num_samples = X.shape[0]
        error_history = []

        for epoch in range(max_epochs):
            total_error = 0

            for i in range(num_samples):
                prediction = self.predict_single(X[i])
                error = y[i] - prediction
                total_error += abs(error)

                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error

            error_history.append(total_error)

            if total_error == 0:
                return epoch + 1, error_history

        return max_epochs, error_history

    def visualize_decision_boundary(self, X, y, title="Perceptron Decision Boundary"):
        # Only works for 2D data
        if X.shape[1] != 2:
            print("Can only visualize 2D data")
            return

        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
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


def generate_linearly_separable_data(n_samples=100, random_state=42):
    """Generate a simple linearly separable dataset.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        X: Feature data
        y: Target labels
    """
    # Generate cluster centers
    centers = [[-1, -1], [1, 1]]

    # Generate blobs with specified centers
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, random_state=random_state, cluster_std=0.7
    )

    return X, y


def generate_nearly_linearly_separable_data(n_samples=100, noise=0.1, random_state=42):
    """Generate data that is nearly linearly separable with some overlap.

    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        random_state: Random seed for reproducibility

    Returns:
        X: Feature data
        y: Target labels
    """
    # Generate a simple dataset with a clear boundary
    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[-1, -1], [1, 1]],
        random_state=random_state,
        cluster_std=0.7,
    )

    # Add noise to create some overlap between classes
    if noise > 0:
        rng = np.random.RandomState(random_state)
        X += rng.normal(0, noise, X.shape)

    return X, y


def generate_non_linearly_separable_data(
    n_samples=100, dataset_type="moons", random_state=42
):
    """Generate data that is not linearly separable.

    Args:
        n_samples: Number of samples to generate
        dataset_type: Type of dataset ('moons' or 'circles')
        random_state: Random seed for reproducibility

    Returns:
        X: Feature data
        y: Target labels
    """
    if dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif dataset_type == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state
        )
    else:
        raise ValueError("dataset_type must be 'moons' or 'circles'")

    return X, y


def experiment_learning_rate(
    X, y, learning_rates=[0.001, 0.01, 0.1, 1.0], max_epochs=100
):
    """
    Experiment with different learning rates to see their effect on convergence.

    Args:
        X: Feature data
        y: Target labels
        learning_rates: List of learning rates to try
        max_epochs: Maximum number of training epochs
    """
    plt.figure(figsize=(12, 8))

    for i, lr in enumerate(learning_rates):
        # Create and train a perceptron with this learning rate
        perceptron = Perceptron(num_features=X.shape[1], learning_rate=lr)
        epochs, error_history = perceptron.train(X, y, max_epochs=max_epochs)

        # Plot the error history
        plt.subplot(2, 2, i + 1)
        plt.plot(range(1, len(error_history) + 1), error_history, marker="o")
        plt.title(f"Learning rate = {lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Number of misclassifications")
        plt.grid(True)

        # Print result
        if error_history[-1] == 0:
            print(f"Learning rate {lr}: Converged in {epochs} epochs")
        else:
            print(f"Learning rate {lr}: Did not converge in {max_epochs} epochs")

    plt.tight_layout()
    plt.show()


def experiment_initialization(X, y, num_trials=5, max_epochs=100):
    """
    Experiment with different random initializations of weights.

    Args:
        X: Feature data
        y: Target labels
        num_trials: Number of different initializations to try
        max_epochs: Maximum number of training epochs
    """
    plt.figure(figsize=(12, 8))

    for i in range(num_trials):
        # Set a different random seed for each initialization
        np.random.seed(i)

        # Create and train a perceptron
        perceptron = Perceptron(num_features=X.shape[1])
        epochs, error_history = perceptron.train(X, y, max_epochs=max_epochs)

        # Plot the error history
        plt.subplot(2, 3, i + 1)
        plt.plot(range(1, len(error_history) + 1), error_history, marker="o")
        plt.title(f"Initialization {i+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Number of misclassifications")
        plt.grid(True)

        # Print result
        print(
            f"Initialization {i+1}: {'Converged' if error_history[-1] == 0 else 'Did not converge'} after {len(error_history)} epochs"
        )

    plt.tight_layout()
    plt.show()


def experiment_different_datasets():
    """
    Experiment with different types of datasets to see where perceptrons succeed and fail.
    """
    # Generate different datasets
    X_linear, y_linear = generate_linearly_separable_data()
    X_noisy, y_noisy = generate_nearly_linearly_separable_data(noise=0.5)
    X_moons, y_moons = generate_non_linearly_separable_data(dataset_type="moons")
    X_circles, y_circles = generate_non_linearly_separable_data(dataset_type="circles")

    # Visualize the datasets
    datasets = [
        (X_linear, y_linear, "Linearly Separable Data"),
        (X_noisy, y_noisy, "Nearly Linearly Separable Data"),
        (X_moons, y_moons, "Non-Linear Data (Moons)"),
        (X_circles, y_circles, "Non-Linear Data (Circles)"),
    ]

    plt.figure(figsize=(14, 10))
    for i, (X, y, title) in enumerate(datasets):
        plt.subplot(2, 2, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o")
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

    # Train perceptrons on each dataset
    for X, y, title in datasets:
        # Create and train a perceptron
        perceptron = Perceptron(num_features=X.shape[1])
        epochs, error_history = perceptron.train(X, y, max_epochs=200)

        # Print results
        if error_history[-1] == 0:
            print(f"\n{title}: Perceptron converged in {epochs} epochs")
        else:
            print(f"\n{title}: Perceptron did not converge in 200 epochs")
            print(f"Final error count: {error_history[-1]} out of {len(y)} samples")

        # Visualize the decision boundary
        perceptron.visualize_decision_boundary(X, y, title=f"Perceptron on {title}")


def main():
    """Main function to run the exercise."""
    print("===== Decision Boundary Visualization Exercise =====")

    # Experiment 1: Different datasets
    print("\n1. Testing perceptron on different datasets")
    experiment_different_datasets()

    # Experiment 2: Learning rate effect
    print("\n2. Effect of learning rate on convergence")
    X, y = generate_linearly_separable_data()
    experiment_learning_rate(X, y)

    # Experiment 3: Initialization effect
    print("\n3. Effect of random initialization on convergence")
    experiment_initialization(X, y)

    print("\nKey takeaways:")
    print("1. Perceptrons can only classify linearly separable data correctly")
    print("2. Learning rate affects convergence speed and stability")
    print("3. Weight initialization can impact whether the perceptron converges at all")


if __name__ == "__main__":
    main()
