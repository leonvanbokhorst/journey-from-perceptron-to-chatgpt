#!/usr/bin/env python3
"""
Classification with MLPs - Exercise 3
Apply MLPs to more complex classification problems and explore mini-batch training
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Layer:
    """
    A fully connected layer in a neural network.
    """

    def __init__(self, n_inputs, n_neurons, activation="relu"):
        """
        Initialize the layer with random weights and zeros for biases.
        """
        # Initialize weights with He initialization (good for ReLU)
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

        # For storing values needed in backpropagation
        self.inputs = None
        self.z = None  # Pre-activation
        self.a = None  # Activation output

        # For gradient storage
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs):
        """
        Perform a forward pass through the layer.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

        # Apply activation function
        if self.activation == "relu":
            self.a = np.maximum(0, self.z)
        elif self.activation == "sigmoid":
            self.a = 1 / (
                1 + np.exp(-np.clip(self.z, -500, 500))
            )  # Clip to avoid overflow
        elif self.activation == "tanh":
            self.a = np.tanh(self.z)
        elif self.activation == "softmax":
            # Numerically stable softmax implementation
            exp_values = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.a = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        elif self.activation == "linear":
            self.a = self.z
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        return self.a

    def backward(self, dvalues):
        """
        Perform a backward pass through the layer.
        """
        # Calculate gradient of activation function
        if self.activation == "relu":
            dactivation = np.where(self.z > 0, 1, 0)
        elif self.activation == "sigmoid":
            dactivation = self.a * (1 - self.a)
        elif self.activation == "tanh":
            dactivation = 1 - self.a**2
        elif self.activation == "softmax":
            # Special case for softmax combined with categorical cross-entropy
            # The gradient is handled differently, and dvalues should already be correct
            dactivation = 1
        elif self.activation == "linear":
            dactivation = 1

        # Gradient on values
        if self.activation == "softmax":
            # For softmax + cross-entropy, dvalues is already the correct gradient
            dz = dvalues
        else:
            dz = dvalues * dactivation

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)

        # Gradient on inputs
        dinputs = np.dot(dz, self.weights.T)

        return dinputs


class MLP:
    """
    Multi-Layer Perceptron for classification tasks.
    """

    def __init__(self, layer_sizes, activations):
        """
        Initialize the MLP with specified layers and activations.

        Args:
            layer_sizes: List of integers denoting the size of each layer
                        (including input and output layers)
            activations: List of activation functions for each layer
                        (length should be one less than layer_sizes)
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                "Number of activation functions must match number of layers - 1"
            )

        self.layers = []

        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            )

    def forward(self, X):
        """
        Perform a forward pass through the network.
        """
        # Forward pass through each layer
        output = X
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def compute_loss(self, y_pred, y_true):
        """
        Compute the loss and its gradient.

        Handles both:
        - Binary cross-entropy for binary classification
        - Categorical cross-entropy for multi-class classification
        """
        n_samples = y_true.shape[0]

        # Multi-class classification
        if y_true.shape[1] > 1:  # One-hot encoded
            # Clip predictions to avoid log(0)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            # Calculate categorical cross-entropy loss
            if len(y_true.shape) == 1:  # Not one-hot, convert to indices
                correct_confidences = y_pred_clipped[range(n_samples), y_true]
            else:  # One-hot encoded
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

            loss = -np.log(correct_confidences)

            # Gradient for softmax + cross-entropy
            dvalues = y_pred.copy()
            dvalues[range(n_samples), np.argmax(y_true, axis=1)] -= 1
            dvalues = dvalues / n_samples

        # Binary classification
        else:
            # Clip predictions to avoid log(0)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            # Calculate binary cross-entropy loss
            loss = -(
                y_true * np.log(y_pred_clipped)
                + (1 - y_true) * np.log(1 - y_pred_clipped)
            )

            # Gradient for sigmoid + binary cross-entropy
            dvalues = (
                -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))
                / n_samples
            )

        return np.mean(loss), dvalues

    def backward(self, dvalues):
        """
        Perform a backward pass through all layers.
        """
        # Start with the gradient from the loss function
        dinputs = dvalues

        # Go through layers in reverse
        for layer in reversed(self.layers):
            dinputs = layer.backward(dinputs)

    def train_batch(self, X_batch, y_batch, learning_rate):
        """
        Train on a single batch of data.
        """
        # Forward pass
        y_pred = self.forward(X_batch)

        # Calculate loss and initial gradient
        loss, dvalues = self.compute_loss(y_pred, y_batch)

        # Backward pass
        self.backward(dvalues)

        # Update parameters
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

        return loss

    def train(
        self,
        X,
        y,
        epochs=1000,
        batch_size=32,
        learning_rate=0.01,
        validation_data=None,
        verbose=True,
    ):
        """
        Train the network using mini-batch gradient descent.
        """
        num_samples = X.shape[0]
        loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            batch_losses = []

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Train on batch
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate)
                batch_losses.append(batch_loss)

            # Calculate average loss for the epoch
            avg_loss = np.mean(batch_losses)
            loss_history.append(avg_loss)

            # Validation if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss, _ = self.compute_loss(y_val_pred, y_val)
                val_loss_history.append(val_loss)

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                val_msg = (
                    f", Validation Loss: {val_loss:.6f}" if validation_data else ""
                )
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}{val_msg}")

        return loss_history, val_loss_history

    def predict(self, X):
        """
        Make predictions for input data X.
        """
        # Forward pass to get outputs
        outputs = self.forward(X)

        # For classification, convert to class predictions
        if outputs.shape[1] > 1:  # Multi-class
            predictions = np.argmax(outputs, axis=1)
        else:  # Binary
            predictions = (outputs > 0.5).astype(int)

        return predictions


def create_synthetic_datasets():
    """
    Create synthetic datasets for classification experiments.
    """
    np.random.seed(42)

    # 1. Two moons dataset (binary classification)
    X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
    y_moons = y_moons.reshape(-1, 1)  # Reshape for binary classification

    # 2. Concentric circles dataset (binary classification)
    X_circles, y_circles = make_circles(
        n_samples=500, noise=0.1, factor=0.5, random_state=42
    )
    y_circles = y_circles.reshape(-1, 1)  # Reshape for binary classification

    # 3. Iris dataset (multi-class classification)
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # One-hot encode the iris targets
    encoder = OneHotEncoder(sparse=False)
    y_iris_onehot = encoder.fit_transform(y_iris.reshape(-1, 1))

    # Split data into train and validation sets
    X_moons_train, X_moons_val, y_moons_train, y_moons_val = train_test_split(
        X_moons, y_moons, test_size=0.2, random_state=42
    )

    X_circles_train, X_circles_val, y_circles_train, y_circles_val = train_test_split(
        X_circles, y_circles, test_size=0.2, random_state=42
    )

    X_iris_train, X_iris_val, y_iris_train, y_iris_val = train_test_split(
        X_iris, y_iris_onehot, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler_moons = StandardScaler()
    X_moons_train = scaler_moons.fit_transform(X_moons_train)
    X_moons_val = scaler_moons.transform(X_moons_val)

    scaler_circles = StandardScaler()
    X_circles_train = scaler_circles.fit_transform(X_circles_train)
    X_circles_val = scaler_circles.transform(X_circles_val)

    scaler_iris = StandardScaler()
    X_iris_train = scaler_iris.fit_transform(X_iris_train)
    X_iris_val = scaler_iris.transform(X_iris_val)

    return {
        "moons": (
            X_moons_train,
            y_moons_train,
            X_moons_val,
            y_moons_val,
            X_moons,
            y_moons,
            scaler_moons,
        ),
        "circles": (
            X_circles_train,
            y_circles_train,
            X_circles_val,
            y_circles_val,
            X_circles,
            y_circles,
            scaler_circles,
        ),
        "iris": (
            X_iris_train,
            y_iris_train,
            X_iris_val,
            y_iris_val,
            X_iris,
            y_iris_onehot,
            scaler_iris,
        ),
    }


def visualize_decision_boundary(model, X, y, scaler, title="Decision Boundary"):
    """
    Visualize decision boundary for 2D data.
    """
    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict class labels for all points in the meshgrid
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.contour(xx, yy, Z, colors="k", linewidths=0.5)

    # Convert y back to integer class for color mapping if it's multi-class
    if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
        y_plot = np.argmax(y, axis=1)
    else:
        y_plot = y.flatten()

    plt.scatter(X[:, 0], X[:, 1], c=y_plot, edgecolors="k", s=40)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def experiment_on_binary_dataset(
    dataset_name, data, architecture_name, layer_sizes, activations
):
    """
    Train and evaluate an MLP on a binary classification dataset.
    """
    X_train, y_train, X_val, y_val, X_full, y_full, scaler = data

    print(f"\nTraining {architecture_name} on {dataset_name} dataset...")

    # Create and train the model
    model = MLP(layer_sizes, activations)
    loss_history, val_loss_history = model.train(
        X_train,
        y_train,
        epochs=1000,
        batch_size=32,
        learning_rate=0.01,
        validation_data=(X_val, y_val),
        verbose=True,
    )

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    accuracy = np.mean(
        y_val_pred == np.argmax(y_val, axis=1)
        if y_val.shape[1] > 1
        else y_val.flatten()
    )
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    # Plot learning curves
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Training Loss")
    if val_loss_history:
        plt.plot(val_loss_history, label="Validation Loss")
    plt.title(f"Learning Curves - {architecture_name} on {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize decision boundary
    if X_full.shape[1] == 2:  # Only for 2D datasets
        visualize_decision_boundary(
            model,
            X_full,
            y_full,
            scaler,
            title=f"{architecture_name} on {dataset_name}",
        )

    return model, accuracy


def experiment_network_architectures():
    """
    Compare different network architectures on the classification datasets.
    """
    # Create datasets
    datasets = create_synthetic_datasets()

    # Define architectures to test
    architectures = [
        {
            "name": "Simple (1 hidden layer)",
            "moons": {"layers": [2, 8, 1], "activations": ["relu", "sigmoid"]},
            "circles": {"layers": [2, 8, 1], "activations": ["relu", "sigmoid"]},
            "iris": {"layers": [4, 8, 3], "activations": ["relu", "softmax"]},
        },
        {
            "name": "Wide (1 wide hidden layer)",
            "moons": {"layers": [2, 32, 1], "activations": ["relu", "sigmoid"]},
            "circles": {"layers": [2, 32, 1], "activations": ["relu", "sigmoid"]},
            "iris": {"layers": [4, 32, 3], "activations": ["relu", "softmax"]},
        },
        {
            "name": "Deep (3 hidden layers)",
            "moons": {
                "layers": [2, 8, 8, 8, 1],
                "activations": ["relu", "relu", "relu", "sigmoid"],
            },
            "circles": {
                "layers": [2, 8, 8, 8, 1],
                "activations": ["relu", "relu", "relu", "sigmoid"],
            },
            "iris": {
                "layers": [4, 8, 8, 8, 3],
                "activations": ["relu", "relu", "relu", "softmax"],
            },
        },
    ]

    # Dictionary to store results
    results = {"moons": [], "circles": [], "iris": []}

    # Run experiments
    for arch in architectures:
        print(f"\nTesting architecture: {arch['name']}")

        # Binary classification - Moons
        _, accuracy_moons = experiment_on_binary_dataset(
            "Moons",
            datasets["moons"],
            arch["name"],
            arch["moons"]["layers"],
            arch["moons"]["activations"],
        )
        results["moons"].append((arch["name"], accuracy_moons))

        # Binary classification - Circles
        _, accuracy_circles = experiment_on_binary_dataset(
            "Circles",
            datasets["circles"],
            arch["name"],
            arch["circles"]["layers"],
            arch["circles"]["activations"],
        )
        results["circles"].append((arch["name"], accuracy_circles))

        # Multi-class classification - Iris
        _, accuracy_iris = experiment_on_binary_dataset(
            "Iris",
            datasets["iris"],
            arch["name"],
            arch["iris"]["layers"],
            arch["iris"]["activations"],
        )
        results["iris"].append((arch["name"], accuracy_iris))

    # Print results summary
    print("\n--- Results Summary ---")
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name.capitalize()} Dataset:")
        print(f"{'Architecture':<20} {'Accuracy':<10}")
        print("-" * 30)
        for arch_name, accuracy in dataset_results:
            print(f"{arch_name:<20} {accuracy * 100:.2f}%")


def experiment_batch_sizes():
    """
    Compare the effect of different batch sizes on training.
    """
    # Create moons dataset
    datasets = create_synthetic_datasets()
    X_train, y_train, X_val, y_val, _, _, _ = datasets["moons"]

    # Batch sizes to test
    batch_sizes = [1, 8, 32, len(X_train)]  # 1 = SGD, full = Batch GD
    batch_names = [
        "SGD (batch=1)",
        "Mini-batch (batch=8)",
        "Mini-batch (batch=32)",
        "Batch GD (full dataset)",
    ]

    # Architecture
    layer_sizes = [2, 16, 1]
    activations = ["relu", "sigmoid"]

    # Results storage
    loss_histories = []
    val_loss_histories = []

    plt.figure(figsize=(10, 6))

    for i, batch_size in enumerate(batch_sizes):
        print(f"\nTraining with batch size: {batch_size}")

        # Create and train model
        model = MLP(layer_sizes, activations)
        loss_history, val_loss_history = model.train(
            X_train,
            y_train,
            epochs=500,  # Fewer epochs for demonstration
            batch_size=batch_size,
            learning_rate=0.01,
            validation_data=(X_val, y_val),
            verbose=False,
        )

        loss_histories.append(loss_history)
        val_loss_histories.append(val_loss_history)

        # Evaluate
        y_val_pred = model.predict(X_val)
        accuracy = np.mean(y_val_pred == y_val.flatten())

        print(f"Final training loss: {loss_history[-1]:.6f}")
        print(f"Final validation loss: {val_loss_history[-1]:.6f}")
        print(f"Validation accuracy: {accuracy * 100:.2f}%")

    # Plot training loss curves
    plt.subplot(1, 2, 1)
    for i, history in enumerate(loss_histories):
        plt.plot(history, label=batch_names[i])
    plt.title("Training Loss by Batch Size")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot validation loss curves
    plt.subplot(1, 2, 2)
    for i, history in enumerate(val_loss_histories):
        plt.plot(history, label=batch_names[i])
    plt.title("Validation Loss by Batch Size")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nKey observations about batch sizes:")
    print("- SGD (batch=1): Noisy updates, can escape local minima, slower convergence")
    print("- Mini-batch: Good balance between update stability and training speed")
    print(
        "- Batch GD: Stable updates but may get stuck in local minima, slower per epoch"
    )


def main():
    """Main function to run the exercise."""
    print("===== Classification with MLPs Exercise =====")

    print("\n1. Comparing Different Network Architectures")
    experiment_network_architectures()

    print("\n2. Exploring the Effect of Batch Size")
    experiment_batch_sizes()

    print("\nThis exercise demonstrated:")
    print("- How to adapt MLPs for different classification tasks")
    print("- The impact of network architecture (width vs depth)")
    print("- The trade-offs with different batch sizes for gradient descent")
    print("- Visualizing decision boundaries for complex datasets")


if __name__ == "__main__":
    main()
