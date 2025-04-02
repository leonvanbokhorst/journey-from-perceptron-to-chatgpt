#!/usr/bin/env python3
"""
RNN with PyTorch - Exercise 4
Implement RNNs using PyTorch's built-in modules
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import time


class SimpleRNN(nn.Module):
    """
    Simple RNN implementation using PyTorch's nn.RNN.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the SimpleRNN model.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            num_layers: Number of stacked RNN layers
        """
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq, features)
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden state (optional)

        Returns:
            output: Prediction
            hn: Final hidden state
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )

        # Forward pass through RNN
        # out shape: (batch_size, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        out, hn = self.rnn(x, h0)

        # Apply output layer to the last output of the RNN sequence
        # Shape: (batch_size, output_size)
        output = self.fc(out[:, -1, :])

        return output, hn


class RNNCell(nn.Module):
    """
    Custom RNN cell implemented using PyTorch's low-level operations.
    This demonstrates how RNNs work at a lower level.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize the RNN cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super(RNNCell, self).__init__()

        # Weight matrices and biases
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev):
        """
        Forward pass through the RNN cell.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)

        Returns:
            h: New hidden state
        """
        # Compute new hidden state
        h = torch.tanh(
            torch.mm(x, self.weight_ih.t())
            + torch.mm(h_prev, self.weight_hh.t())
            + self.bias
        )

        return h


class ManualRNN(nn.Module):
    """
    Manual implementation of an RNN using our custom RNN cell.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the ManualRNN model.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
        """
        super(ManualRNN, self).__init__()

        self.hidden_size = hidden_size

        # RNN cell
        self.rnn_cell = RNNCell(input_size, hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Initial hidden state (optional)

        Returns:
            output: Prediction
            hidden_states: List of all hidden states
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process sequence step by step
        h = h0
        hidden_states = [h]

        for t in range(seq_len):
            # Get input at current time step
            xt = x[:, t, :]

            # Update hidden state
            h = self.rnn_cell(xt, h)

            # Store hidden state
            hidden_states.append(h)

        # Apply output layer to the last hidden state
        output = self.fc(hidden_states[-1])

        return output, hidden_states


class LSTMModel(nn.Module):
    """
    LSTM implementation using PyTorch's nn.LSTM module.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the LSTM model.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            num_layers: Number of stacked LSTM layers
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Initial hidden state and cell state (optional)

        Returns:
            output: Prediction
            (hn, cn): Final hidden state and cell state
        """
        batch_size = x.size(0)

        # Initialize hidden and cell states if not provided
        if hidden is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )
            c0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )
            hidden = (h0, c0)

        # Forward pass through LSTM
        # out shape: (batch_size, seq_len, hidden_size)
        # hn, cn shape: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, hidden)

        # Apply output layer to the last output of the LSTM sequence
        output = self.fc(out[:, -1, :])

        return output, (hn, cn)


def generate_sine_wave():
    """
    Generate a sine wave dataset for demonstration.

    Returns:
        X_train: Training input sequences
        y_train: Training target values
        X_test: Test input sequences
        y_test: Test target values
        scaler: Fitted MinMaxScaler for inverse transformation
    """
    # Generate a sine wave
    time_steps = np.linspace(0, 10, 1000)
    data = np.sin(time_steps)

    # Add some noise
    data = data + 0.1 * np.random.randn(len(time_steps))

    # Scale data to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    # Create sequences
    seq_length = 20
    X, y = [], []

    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i : i + seq_length])
        y.append(data_scaled[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    return X_train, y_train, X_test, y_test, scaler


def train_model(
    model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.01
):
    """
    Train a PyTorch RNN model.

    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Size of batches for training
        lr: Learning rate

    Returns:
        model: Trained model
        train_losses: List of training losses
        test_losses: List of test losses
    """
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For tracking losses
    train_losses = []
    test_losses = []

    # Start timer
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        model.train()

        train_loss = 0
        for batch_X, batch_y in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average training loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.6f}, "
                f"Test Loss: {test_loss:.6f}"
            )

    # End timer
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    return model, train_losses, test_losses


def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare different RNN implementations on the same task.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    """
    # Define models to compare
    input_size = 1
    hidden_size = 16
    output_size = 1

    models = {
        "PyTorch RNN": SimpleRNN(input_size, hidden_size, output_size),
        "Manual RNN": ManualRNN(input_size, hidden_size, output_size),
        "PyTorch LSTM": LSTMModel(input_size, hidden_size, output_size),
    }

    # Training parameters
    epochs = 100
    batch_size = 32
    lr = 0.01

    # Results
    all_train_losses = {}
    all_test_losses = {}
    training_times = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        start_time = time.time()

        # Train the model
        _, train_losses, test_losses = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        end_time = time.time()
        training_time = end_time - start_time

        # Store results
        all_train_losses[name] = train_losses
        all_test_losses[name] = test_losses
        training_times[name] = training_time

    # Plot training losses
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for name, losses in all_train_losses.items():
        plt.plot(losses, label=name)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, losses in all_test_losses.items():
        plt.plot(losses, label=name)
    plt.title("Test Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nModel Comparison Summary:")
    print(
        f"{'Model':<15} {'Final Train Loss':<20} {'Final Test Loss':<20} {'Training Time (s)':<20}"
    )
    print("-" * 75)

    for name in models.keys():
        print(
            f"{name:<15} {all_train_losses[name][-1]:<20.6f} {all_test_losses[name][-1]:<20.6f} {training_times[name]:<20.2f}"
        )


def make_predictions(model, X_test, y_test, scaler, n_future=50):
    """
    Make predictions using the trained model and visualize them.

    Args:
        model: Trained PyTorch model
        X_test, y_test: Test data
        scaler: Scaler to transform data back to original scale
        n_future: Number of future points to predict
    """
    model.eval()

    # Make predictions on test data
    with torch.no_grad():
        y_pred, _ = model(X_test)

    # Convert to numpy for easier manipulation
    y_pred = y_pred.numpy()
    y_test = y_test.numpy()

    # Inverse transform to original scale
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test)

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig, label="Actual")
    plt.plot(y_pred_orig, label="Predicted")
    plt.title("Model Predictions vs Actual Values")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Generate future predictions
    # Start with the last sequence from X_test
    current_seq = X_test[-1].unsqueeze(0)  # Add batch dimension
    future_preds = []

    # Generate future predictions one step at a time
    for _ in range(n_future):
        # Predict next step
        pred, _ = model(current_seq)
        future_preds.append(pred.item())

        # Update sequence: remove first element, add prediction
        current_seq = torch.cat(
            [current_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)], dim=1
        )

    # Inverse transform future predictions
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Plot continuous prediction
    plt.figure(figsize=(12, 6))

    # Last 100 actual points
    last_actual = y_test_orig[-100:].flatten()
    plt.plot(range(len(last_actual)), last_actual, "b-", label="Actual")

    # Future predictions
    plt.plot(
        range(len(last_actual) - 1, len(last_actual) + n_future),
        np.vstack([y_test_orig[-1], future_preds]).flatten(),
        "r--",
        label="Future Predictions",
    )

    plt.axvline(x=len(last_actual) - 1, color="k", linestyle="-", alpha=0.2)
    plt.title("Future Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def explore_hyperparameters():
    """
    Explore the impact of different hyperparameters on model performance.
    """
    # Generate data
    X_train, y_train, X_test, y_test, scaler = generate_sine_wave()

    # Hyperparameters to explore
    hidden_sizes = [8, 16, 32, 64]
    learning_rates = [0.001, 0.01, 0.1]

    # Results
    results = []

    # Base model configuration
    input_size = 1
    output_size = 1
    epochs = 100
    batch_size = 32

    # Test different hidden sizes
    print("\nExploring different hidden sizes...")
    for hidden_size in hidden_sizes:
        model = SimpleRNN(input_size, hidden_size, output_size)

        _, _, test_losses = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=0.01,
        )

        final_loss = test_losses[-1]
        results.append(("Hidden Size", hidden_size, final_loss))

    # Test different learning rates
    print("\nExploring different learning rates...")
    for lr in learning_rates:
        model = SimpleRNN(input_size, 16, output_size)

        _, _, test_losses = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

        final_loss = test_losses[-1]
        results.append(("Learning Rate", lr, final_loss))

    # Plot results
    plt.figure(figsize=(12, 6))

    # Hidden sizes
    hidden_size_results = [(r[1], r[2]) for r in results if r[0] == "Hidden Size"]
    plt.subplot(1, 2, 1)
    plt.bar(
        [str(hs) for hs, _ in hidden_size_results],
        [loss for _, loss in hidden_size_results],
    )
    plt.title("Effect of Hidden Size on Loss")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final Test Loss")
    plt.grid(True)

    # Learning rates
    lr_results = [(r[1], r[2]) for r in results if r[0] == "Learning Rate"]
    plt.subplot(1, 2, 2)
    plt.bar([str(lr) for lr, _ in lr_results], [loss for _, loss in lr_results])
    plt.title("Effect of Learning Rate on Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Test Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nHyperparameter Exploration Summary:")
    print("\nHidden Size Results:")
    print(f"{'Hidden Size':<15} {'Final Test Loss':<20}")
    print("-" * 35)
    for hs, loss in hidden_size_results:
        print(f"{hs:<15} {loss:<20.6f}")

    print("\nLearning Rate Results:")
    print(f"{'Learning Rate':<15} {'Final Test Loss':<20}")
    print("-" * 35)
    for lr, loss in lr_results:
        print(f"{lr:<15} {loss:<20.6f}")


def visualize_hidden_states():
    """
    Visualize the hidden states of an RNN to understand how information flows.
    """
    # Generate simple data
    X_train, y_train, X_test, y_test, _ = generate_sine_wave()

    # Create a simple RNN and manual RNN for comparison
    input_size = 1
    hidden_size = 8
    output_size = 1

    manual_rnn = ManualRNN(input_size, hidden_size, output_size)

    # Train manual RNN for a few epochs
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(manual_rnn.parameters(), lr=0.01)

    for epoch in range(10):
        manual_rnn.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs, _ = manual_rnn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Get hidden states for a test sequence
    sample_idx = 0
    input_seq = X_test[sample_idx].unsqueeze(0)  # Add batch dimension

    manual_rnn.eval()
    with torch.no_grad():
        _, hidden_states = manual_rnn(input_seq)

    # Convert hidden states to numpy for visualization
    hidden_np = torch.stack(hidden_states).squeeze(1).numpy()

    # Plot input sequence
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(input_seq.squeeze().numpy())
    plt.title("Input Sequence")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)

    # Plot hidden states as a heatmap
    plt.subplot(2, 1, 2)
    plt.imshow(hidden_np.T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Hidden State Value")
    plt.title("Hidden State Evolution")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Unit")
    plt.tight_layout()
    plt.show()

    # Plot selected hidden units over time
    plt.figure(figsize=(12, 6))

    for i in range(min(4, hidden_size)):
        plt.plot(hidden_np[:, i], label=f"Hidden Unit {i+1}")

    plt.title("Selected Hidden Units Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden State Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the PyTorch RNN exercise."""
    print("===== RNN with PyTorch Exercise =====")

    # 1. Generate data
    print("\n1. Generating data...")
    X_train, y_train, X_test, y_test, scaler = generate_sine_wave()

    # Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 2. Compare different models
    print("\n2. Comparing different RNN implementations...")
    compare_models(X_train, y_train, X_test, y_test)

    # 3. Make predictions with the best model
    print("\n3. Making predictions with PyTorch LSTM...")
    lstm_model = LSTMModel(input_size=1, hidden_size=16, output_size=1)

    # Train the model
    lstm_model, _, _ = train_model(
        lstm_model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.01
    )

    # Make and visualize predictions
    make_predictions(lstm_model, X_test, y_test, scaler, n_future=100)

    # 4. Explore hyperparameters
    print("\n4. Exploring hyperparameters...")
    explore_hyperparameters()

    # 5. Visualize hidden states
    print("\n5. Visualizing hidden states...")
    visualize_hidden_states()

    print("\nThis exercise demonstrated:")
    print("- How to implement RNNs using PyTorch's built-in modules")
    print("- The difference between PyTorch's high-level and manual implementations")
    print("- The advantage of using LSTM for sequence modeling")
    print("- How to make predictions and visualize results")
    print("- The impact of hyperparameters on model performance")


if __name__ == "__main__":
    main()
