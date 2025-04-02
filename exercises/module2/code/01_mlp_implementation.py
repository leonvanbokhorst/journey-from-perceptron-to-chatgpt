#!/usr/bin/env python3
"""
MLP Implementation - Exercise 1
Build a multi-layer perceptron from scratch with backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt


class Layer:
    """
    A fully connected layer in a neural network.
    """

    def __init__(self, n_inputs, n_neurons, activation="sigmoid"):
        """
        Initialize the layer with random weights and zeros for biases.

        Args:
            n_inputs: Number of inputs to the layer
            n_neurons: Number of neurons in the layer
            activation: Activation function to use ('sigmoid', 'tanh', or 'relu')
        """
        # Initialize weights with small random values (scaled by sqrt of n_inputs)
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        # Initialize biases with zeros
        self.biases = np.zeros((1, n_neurons))
        # Set activation function
        self.activation = activation

        # Storage for forward pass values (needed for backpropagation)
        self.inputs = None
        self.z = None  # Weighted inputs
        self.a = None  # Activations

    def forward(self, inputs):
        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data (batch_size, n_inputs)

        Returns:
            Activations of the layer (batch_size, n_neurons)
        """
        # Store inputs for backpropagation
        self.inputs = inputs

        # Calculate weighted inputs
        self.z = np.dot(inputs, self.weights) + self.biases

        # Apply activation function
        if self.activation == "sigmoid":
            self.a = 1 / (1 + np.exp(-self.z))
        elif self.activation == "tanh":
            self.a = np.tanh(self.z)
        elif self.activation == "relu":
            self.a = np.maximum(0, self.z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        return self.a

    def backward(self, dvalues):
        """
        Perform a backward pass through the layer.

        Args:
            dvalues: Gradients from the next layer (batch_size, n_neurons)

        Returns:
            Gradients for the inputs to this layer
        """
        # Calculate gradient of activation function
        if self.activation == "sigmoid":
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            dactivation = self.a * (1 - self.a)
        elif self.activation == "tanh":
            # Derivative of tanh: 1 - tanh(x)^2
            dactivation = 1 - self.a**2
        elif self.activation == "relu":
            # Derivative of ReLU: 1 if x > 0 else 0
            dactivation = np.where(self.z > 0, 1, 0)

        # Multiply incoming gradients by derivatives (chain rule)
        dz = dvalues * dactivation

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)

        # Gradients on inputs to this layer (for propagating backwards)
        dinputs = np.dot(dz, self.weights.T)

        return dinputs


class MLP:
    """
    Multi-Layer Perceptron implementation with backpropagation.
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

    def forward(self, inputs):
        """
        Perform a forward pass through the network.

        Args:
            inputs: Input data (batch_size, n_features)

        Returns:
            Output of the network
        """
        # Current inputs become the input to the first layer
        x = inputs

        # Forward pass through each layer
        for layer in self.layers:
            x = layer.forward(x)

        # Return the output
        return x

    def backward(self, y_pred, y_true):
        """
        Perform a backward pass through the network.

        Args:
            y_pred: Predictions from the network
            y_true: True values

        Returns:
            Dictionary with gradients for each layer
        """
        # Calculate loss gradient (assuming MSE loss)
        # For MSE: dL/dy_pred = 2 * (y_pred - y_true) / n_samples
        n_samples = y_true.shape[0]
        dvalues = 2 * (y_pred - y_true) / n_samples

        # Backward pass through all layers in reverse order
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def train_step(self, X, y, learning_rate):
        """
        Perform a single training step (forward pass, backward pass, update).

        Args:
            X: Training data
            y: Target values
            learning_rate: Learning rate for parameter updates

        Returns:
            Loss value
        """
        # Forward pass
        y_pred = self.forward(X)

        # Calculate loss (mean squared error)
        loss = np.mean((y_pred - y) ** 2)

        # Backward pass (calculate gradients)
        self.backward(y_pred, y)

        # Update parameters
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

        return loss

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """
        Train the network using gradient descent.

        Args:
            X: Training data
            y: Target values
            epochs: Number of training epochs
            learning_rate: Learning rate for parameter updates
            verbose: Whether to print progress during training

        Returns:
            List of loss values during training
        """
        loss_history = []

        for epoch in range(epochs):
            # Perform training step
            loss = self.train_step(X, y, learning_rate)
            loss_history.append(loss)

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return loss_history

    def predict(self, X):
        """
        Make predictions for input data X.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        return self.forward(X)


def visualize_decision_boundary(model, X, y, title="MLP Decision Boundary"):
    """
    Visualize the decision boundary of a model for 2D data.

    Args:
        model: Model with predict method
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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Make predictions on the mesh grid
    grid_inputs = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_inputs)

    # For binary classification, threshold at 0.5
    if Z.shape[1] == 1:
        Z = (Z > 0.5).astype(float)
    else:
        Z = np.argmax(Z, axis=1)

    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and training points
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.contour(xx, yy, Z, colors="k", linewidths=1)

    # If y is 2D (one-hot encoded), convert to 1D for plotting
    if y.ndim > 1 and y.shape[1] > 1:
        y_plot = np.argmax(y, axis=1)
    else:
        y_plot = y

    plt.scatter(X[:, 0], X[:, 1], c=y_plot, edgecolors="k", marker="o", s=100)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def solve_xor():
    """
    Train an MLP to solve the XOR problem and visualize the results.
    """
    # Define the XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR outputs

    # Create an MLP with one hidden layer (2 input -> 4 hidden -> 1 output)
    model = MLP([2, 4, 1], activations=["sigmoid", "sigmoid"])

    # Train the model
    print("Training MLP for XOR problem...")
    loss_history = model.train(X, y, epochs=5000, learning_rate=0.1)

    # Plot the loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title("Learning Curve for XOR")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize the decision boundary
    visualize_decision_boundary(model, X, y, title="MLP Solution for XOR")

    # Test the model
    y_pred = model.predict(X)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\nTesting the trained MLP:")
    print("Inputs | Target | Prediction | Rounded")
    print("-------+--------+------------+--------")
    for i in range(len(X)):
        print(
            f"{X[i][0]}, {X[i][1]}   |   {y[i][0]}    |   {y_pred[i][0]:.4f}   |   {y_pred_binary[i][0]}"
        )


def explore_network_depth():
    """
    Explore how network depth affects learning by comparing different architectures.
    """
    # Define the XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR outputs

    # Define different network architectures
    architectures = [
        {
            "name": "No Hidden Layer (Perceptron)",
            "layers": [2, 1],
            "activations": ["sigmoid"],
        },
        {
            "name": "1 Hidden Layer (2 neurons)",
            "layers": [2, 2, 1],
            "activations": ["sigmoid", "sigmoid"],
        },
        {
            "name": "1 Hidden Layer (4 neurons)",
            "layers": [2, 4, 1],
            "activations": ["sigmoid", "sigmoid"],
        },
        {
            "name": "2 Hidden Layers",
            "layers": [2, 3, 2, 1],
            "activations": ["sigmoid", "sigmoid", "sigmoid"],
        },
    ]

    plt.figure(figsize=(12, 8))

    # Train each architecture and plot the learning curves
    for i, arch in enumerate(architectures):
        # Create the model
        model = MLP(arch["layers"], arch["activations"])

        # Train the model
        print(f"\nTraining {arch['name']}...")
        loss_history = model.train(X, y, epochs=2000, learning_rate=0.1, verbose=False)

        # Plot the learning curve
        plt.subplot(2, 2, i + 1)
        plt.plot(loss_history)
        plt.title(arch["name"])
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.grid(True)

        # Make predictions and calculate accuracy
        y_pred = model.predict(X)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred_binary == y) * 100

        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Accuracy: {accuracy:.1f}%")

    plt.tight_layout()
    plt.show()

    print("\nObservations:")
    print("1. The single-layer perceptron cannot solve XOR (non-linearly separable).")
    print("2. Adding a hidden layer allows the network to solve XOR.")
    print("3. Increasing the width (more neurons) can help learning.")
    print("4. Deeper networks may learn faster but can be harder to train properly.")


def main():
    """Main function to run the exercise."""
    print("===== Multi-Layer Perceptron Implementation Exercise =====")

    print("\n1. Solving XOR Problem with MLP")
    solve_xor()

    print("\n2. Exploring Network Depth and Width")
    explore_network_depth()

    print("\nThis exercise demonstrated:")
    print("- How to implement a neural network from scratch using NumPy")
    print("- The forward and backward pass algorithms")
    print("- How MLPs can solve problems that perceptrons cannot (like XOR)")
    print("- The effect of different network architectures on learning")

    print("\nIn the next exercise, we'll explore different activation functions!")


if __name__ == "__main__":
    main()
