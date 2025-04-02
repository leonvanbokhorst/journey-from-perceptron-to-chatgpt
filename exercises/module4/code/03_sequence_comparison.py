"""
Sequence Prediction with LSTM and GRU

This exercise compares the performance of standard RNNs, LSTMs, and GRUs
on sequence prediction tasks of varying complexity and length.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict, Optional, Any

# Import models from previous exercises
import sys
import os

# Add parent directory to path to import from previous exercises if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Basic RNN implementation
class SimpleRNN:
    """
    A simple Recurrent Neural Network implementation for comparison.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a simple RNN.

        Args:
            input_size: Size of the input vector
            hidden_size: Size of the hidden state vector
            output_size: Size of the output vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, input_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(output_size, hidden_size) * scale

        # Bias initialization
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Memory for storing states and activations
        self.h_states = {}
        self.outputs = {}

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """
        Forward pass through the RNN for the entire sequence.

        Args:
            x_sequence: Input sequence of shape (sequence_length, input_size)

        Returns:
            y_sequence: Output sequence of shape (sequence_length, output_size)
        """
        sequence_length = len(x_sequence)

        # Initialize hidden state
        h_prev = np.zeros((self.hidden_size, 1))

        # Process each time step
        y_sequence = np.zeros((sequence_length, self.output_size))

        for t in range(sequence_length):
            # Get input at current time step
            x_t = x_sequence[t].reshape(-1, 1)

            # Update hidden state
            h_t = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, h_prev) + self.bh)

            # Compute output
            y_t = np.dot(self.Why, h_t) + self.by

            # Store states and outputs for backpropagation
            self.h_states[t] = h_t
            self.outputs[t] = y_t

            # Update for next time step
            h_prev = h_t

            # Store output
            y_sequence[t] = y_t.flatten()

        return y_sequence

    def backward(self, dy_sequence: np.ndarray, learning_rate: float = 0.01) -> None:
        """
        Backward pass through the RNN for the entire sequence.

        Args:
            dy_sequence: Gradient of the loss with respect to the outputs
            learning_rate: Learning rate for gradient descent
        """
        sequence_length = len(dy_sequence)

        # Initialize parameter gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Initialize gradient for the last hidden state
        dh_next = np.zeros((self.hidden_size, 1))

        # Process gradients backward in time
        for t in reversed(range(sequence_length)):
            # Get gradient for current time step
            dy_t = dy_sequence[t].reshape(-1, 1)

            # Gradient for output layer
            dWhy += np.dot(dy_t, self.h_states[t].T)
            dby += dy_t

            # Gradient for hidden state
            dh = np.dot(self.Why.T, dy_t) + dh_next

            # Gradient through tanh
            dh_raw = (1 - self.h_states[t] ** 2) * dh

            # Accumulate parameter gradients
            dbh += dh_raw

            # Get previous hidden state and input
            h_prev = np.zeros((self.hidden_size, 1)) if t == 0 else self.h_states[t - 1]
            x_t = (
                np.zeros((self.input_size, 1))
                if t == 0
                else np.array(np.reshape(dy_sequence[t - 1], (self.input_size, 1)))
            )

            dWxh += np.dot(dh_raw, x_t.T)
            dWhh += np.dot(dh_raw, h_prev.T)

            # Gradient for next iteration
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update parameters with gradients
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        print_every: int = 10,
    ) -> List[float]:
        """
        Train the RNN on sequence data.

        Args:
            X: Training data of shape (num_sequences, sequence_length, input_size)
            y: Target data of shape (num_sequences, sequence_length, output_size)
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            print_every: Print loss every print_every epochs

        Returns:
            losses: List of losses during training
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            for i in range(len(X)):
                # Forward pass
                y_pred = self.forward(X[i])

                # Compute MSE loss
                loss = np.mean((y_pred - y[i]) ** 2)
                epoch_loss += loss

                # Compute gradient of loss
                dy = 2 * (y_pred - y[i]) / y_pred.size

                # Backward pass
                self.backward(dy, learning_rate)

            epoch_loss /= len(X)
            losses.append(epoch_loss)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        return losses


# Import LSTM and GRU from previous exercises
try:
    import importlib.util
    import os

    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Import LSTM
    lstm_path = os.path.join(current_dir, "01_lstm_implementation.py")
    lstm_spec = importlib.util.spec_from_file_location("lstm_module", lstm_path)
    lstm_module = importlib.util.module_from_spec(lstm_spec)
    lstm_spec.loader.exec_module(lstm_module)
    LSTM = lstm_module.LSTM

    # Import GRU
    gru_path = os.path.join(current_dir, "02_gru_implementation.py")
    gru_spec = importlib.util.spec_from_file_location("gru_module", gru_path)
    gru_module = importlib.util.module_from_spec(gru_spec)
    gru_spec.loader.exec_module(gru_module)
    GRU = gru_module.GRU
except ImportError:
    print("Warning: Could not import LSTM or GRU from previous exercises.")
    print("Implementing simplified versions for comparison.")

    # Simplified LSTM for comparison
    class LSTM(SimpleRNN):
        """Simplified LSTM implementation for comparison."""

        pass

    # Simplified GRU for comparison
    class GRU(SimpleRNN):
        """Simplified GRU implementation for comparison."""

        pass


def generate_sine_wave(
    samples: int = 100, period: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a sine wave dataset for sequence prediction.

    Args:
        samples: Number of samples
        period: Period of the sine wave

    Returns:
        X: Input sequences
        y: Target sequences (next values in the sequence)
    """
    # Generate sine wave
    x = np.linspace(0, 10 * np.pi, samples)
    sine_wave = np.sin(x)

    # Create sequences for training
    X = []
    y = []

    for i in range(len(sine_wave) - period - 1):
        X.append(sine_wave[i : i + period])
        y.append(sine_wave[i + 1 : i + period + 1])

    return np.array(X), np.array(y)


def generate_long_term_dependency_data(
    sequence_length: int = 100, num_sequences: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data with long-term dependencies where the output depends on
    both recent and distant past inputs.

    Args:
        sequence_length: Length of each sequence
        num_sequences: Number of sequences to generate

    Returns:
        X: Input sequences
        y: Target sequences
    """
    X = np.random.rand(num_sequences, sequence_length, 1)
    y = np.zeros_like(X)

    for i in range(num_sequences):
        # Output depends on the product of the first and current input
        for t in range(sequence_length):
            y[i, t, 0] = X[i, 0, 0] * X[i, t, 0]

    return X, y


def compare_models_on_sine_wave() -> Dict[str, List[float]]:
    """
    Compare RNN, LSTM, and GRU on sine wave prediction.

    Returns:
        Dict containing loss history for each model
    """
    # Generate data
    X, y = generate_sine_wave(samples=1000, period=20)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for model input (sequence_length, input_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    # Initialize models
    rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
    lstm = LSTM(input_size=1, hidden_size=10, output_size=1)
    gru = GRU(input_size=1, hidden_size=10, output_size=1)

    # Train models
    print("Training Simple RNN...")
    rnn_losses = rnn.train(
        X_train, y_train, epochs=50, learning_rate=0.01, print_every=10
    )

    print("\nTraining LSTM...")
    lstm_losses = lstm.train(
        X_train, y_train, epochs=50, learning_rate=0.01, print_every=10
    )

    print("\nTraining GRU...")
    gru_losses = gru.train(
        X_train, y_train, epochs=50, learning_rate=0.01, print_every=10
    )

    # Evaluate on test set
    rnn_test_loss = 0
    lstm_test_loss = 0
    gru_test_loss = 0

    for i in range(len(X_test)):
        rnn_pred = rnn.forward(X_test[i])
        lstm_pred = lstm.forward(X_test[i])
        gru_pred = gru.forward(X_test[i])

        rnn_test_loss += np.mean((rnn_pred - y_test[i].reshape(-1)) ** 2)
        lstm_test_loss += np.mean((lstm_pred - y_test[i].reshape(-1)) ** 2)
        gru_test_loss += np.mean((gru_pred - y_test[i].reshape(-1)) ** 2)

    rnn_test_loss /= len(X_test)
    lstm_test_loss /= len(X_test)
    gru_test_loss /= len(X_test)

    print(
        f"\nTest loss - RNN: {rnn_test_loss:.4f}, LSTM: {lstm_test_loss:.4f}, GRU: {gru_test_loss:.4f}"
    )

    # Plot comparison of training losses
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label="RNN")
    plt.plot(lstm_losses, label="LSTM")
    plt.plot(gru_losses, label="GRU")
    plt.title("Training Loss Comparison on Sine Wave Prediction")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize predictions
    idx = np.random.randint(0, len(X_test))
    X_sample = X_test[idx]
    y_sample = y_test[idx]

    rnn_pred = rnn.forward(X_sample)
    lstm_pred = lstm.forward(X_sample)
    gru_pred = gru.forward(X_sample)

    plt.figure(figsize=(12, 6))
    plt.plot(X_sample, "b", label="Input", alpha=0.3)
    plt.plot(y_sample, "g", label="Target")
    plt.plot(rnn_pred, "r--", label="RNN")
    plt.plot(lstm_pred, "m--", label="LSTM")
    plt.plot(gru_pred, "c--", label="GRU")
    plt.legend()
    plt.title("Sine Wave Prediction Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    return {"rnn": rnn_losses, "lstm": lstm_losses, "gru": gru_losses}


def compare_models_on_long_term_dependency() -> Dict[str, List[float]]:
    """
    Compare RNN, LSTM, and GRU on data with long-term dependencies.

    Returns:
        Dict containing loss history for each model
    """
    # Generate data
    X, y = generate_long_term_dependency_data(sequence_length=50, num_sequences=200)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize models
    rnn = SimpleRNN(input_size=1, hidden_size=20, output_size=1)
    lstm = LSTM(input_size=1, hidden_size=20, output_size=1)
    gru = GRU(input_size=1, hidden_size=20, output_size=1)

    # Train models
    print("Training Simple RNN on long-term dependency data...")
    rnn_losses = rnn.train(
        X_train, y_train, epochs=100, learning_rate=0.01, print_every=10
    )

    print("\nTraining LSTM on long-term dependency data...")
    lstm_losses = lstm.train(
        X_train, y_train, epochs=100, learning_rate=0.01, print_every=10
    )

    print("\nTraining GRU on long-term dependency data...")
    gru_losses = gru.train(
        X_train, y_train, epochs=100, learning_rate=0.01, print_every=10
    )

    # Evaluate on test set
    rnn_test_loss = 0
    lstm_test_loss = 0
    gru_test_loss = 0

    for i in range(len(X_test)):
        rnn_pred = rnn.forward(X_test[i])
        lstm_pred = lstm.forward(X_test[i])
        gru_pred = gru.forward(X_test[i])

        rnn_test_loss += np.mean((rnn_pred - y_test[i].reshape(-1)) ** 2)
        lstm_test_loss += np.mean((lstm_pred - y_test[i].reshape(-1)) ** 2)
        gru_test_loss += np.mean((gru_pred - y_test[i].reshape(-1)) ** 2)

    rnn_test_loss /= len(X_test)
    lstm_test_loss /= len(X_test)
    gru_test_loss /= len(X_test)

    print(
        f"\nTest loss on long-term dependency - RNN: {rnn_test_loss:.4f}, LSTM: {lstm_test_loss:.4f}, GRU: {gru_test_loss:.4f}"
    )

    # Plot comparison of training losses
    plt.figure(figsize=(10, 6))
    plt.plot(rnn_losses, label="RNN")
    plt.plot(lstm_losses, label="LSTM")
    plt.plot(gru_losses, label="GRU")
    plt.title("Training Loss Comparison on Long-Term Dependency Task")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize predictions for a sample sequence
    idx = np.random.randint(0, len(X_test))
    X_sample = X_test[idx]
    y_sample = y_test[idx]

    rnn_pred = rnn.forward(X_sample)
    lstm_pred = lstm.forward(X_sample)
    gru_pred = gru.forward(X_sample)

    plt.figure(figsize=(12, 6))
    plt.plot(y_sample, "g", label="Target")
    plt.plot(rnn_pred, "r--", label="RNN")
    plt.plot(lstm_pred, "m--", label="LSTM")
    plt.plot(gru_pred, "c--", label="GRU")

    # Mark the first input value that influences all outputs
    plt.axhline(
        y=X_sample[0, 0],
        color="b",
        linestyle="--",
        alpha=0.5,
        label=f"First Input: {X_sample[0, 0]:.2f}",
    )

    plt.legend()
    plt.title("Long-Term Dependency Task Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    return {"rnn": rnn_losses, "lstm": lstm_losses, "gru": gru_losses}


def compare_training_speed() -> Dict[str, float]:
    """
    Compare the training speed of RNN, LSTM, and GRU models.

    Returns:
        Dict containing training time for each model
    """
    # Generate data
    X, y = generate_sine_wave(samples=500, period=20)

    # Reshape for model input
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    # Initialize models
    rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
    lstm = LSTM(input_size=1, hidden_size=10, output_size=1)
    gru = GRU(input_size=1, hidden_size=10, output_size=1)

    # Time RNN training
    print("Timing RNN training...")
    start_time = time.time()
    rnn.train(X, y, epochs=20, print_every=20)
    rnn_time = time.time() - start_time

    # Time LSTM training
    print("\nTiming LSTM training...")
    start_time = time.time()
    lstm.train(X, y, epochs=20, print_every=20)
    lstm_time = time.time() - start_time

    # Time GRU training
    print("\nTiming GRU training...")
    start_time = time.time()
    gru.train(X, y, epochs=20, print_every=20)
    gru_time = time.time() - start_time

    # Print results
    print("\nTraining time comparison:")
    print(f"RNN: {rnn_time:.2f} seconds")
    print(f"LSTM: {lstm_time:.2f} seconds")
    print(f"GRU: {gru_time:.2f} seconds")

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(["RNN", "LSTM", "GRU"], [rnn_time, lstm_time, gru_time])
    plt.title("Training Time Comparison (20 epochs)")
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.grid(True, axis="y")
    plt.show()

    return {"rnn": rnn_time, "lstm": lstm_time, "gru": gru_time}


def visualize_gradient_flow(epochs: int = 20) -> None:
    """
    Visualize how gradients flow through RNN, LSTM, and GRU during training,
    demonstrating the vanishing gradient problem in standard RNNs.

    Args:
        epochs: Number of epochs to train for gradient visualization
    """
    # Generate a simple dataset with very long sequences
    sequence_length = 100
    num_sequences = 50

    # Create data where the output is the first element of the sequence
    X = np.random.rand(num_sequences, sequence_length, 1)
    y = np.zeros_like(X)

    for i in range(num_sequences):
        y[i, :, 0] = X[i, 0, 0]  # Output is always the first element

    # Initialize models
    rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
    lstm = LSTM(input_size=1, hidden_size=5, output_size=1)
    gru = GRU(input_size=1, hidden_size=5, output_size=1)

    # Track gradients at different time steps
    gradient_norms = {
        "rnn": np.zeros((epochs, sequence_length)),
        "lstm": np.zeros((epochs, sequence_length)),
        "gru": np.zeros((epochs, sequence_length)),
    }

    # Function to compute and record gradient norms
    def record_gradients(model_name, model, epoch):
        for t in range(sequence_length):
            # Forward and backward through a single example
            sample_idx = np.random.randint(0, num_sequences)
            _ = model.forward(X[sample_idx])

            # Compute gradient at this time step (simplified)
            dy = np.zeros((sequence_length, 1))
            dy[-1] = 1.0  # Gradient at the last time step

            if model_name == "rnn":
                # Compute gradients through time for RNN
                dh_next = np.zeros((model.hidden_size, 1))

                for i in reversed(range(sequence_length)):
                    dh = np.dot(model.Why.T, dy[i].reshape(-1, 1)) + dh_next
                    dh_raw = (1 - model.h_states[i] ** 2) * dh

                    # Gradient norm at this time step
                    gradient_norms[model_name][epoch, i] = np.linalg.norm(dh_raw)

                    # Propagate gradient to next iteration
                    h_prev = (
                        np.zeros((model.hidden_size, 1))
                        if i == 0
                        else model.h_states[i - 1]
                    )
                    dh_next = np.dot(model.Whh.T, dh_raw)

            else:
                # For LSTM and GRU, just use a simplified gradient flow calculation
                # This is a simplification as the actual backprop is more complex

                # Initialize gradients
                if model_name == "lstm":
                    dh_next = np.zeros((model.hidden_size, 1))
                    dc_next = np.zeros((model.hidden_size, 1))

                    for i in reversed(range(sequence_length)):
                        dh = np.zeros((model.hidden_size, 1))
                        dh += np.dot(model.Wy.T, dy[i].reshape(-1, 1)) + dh_next

                        # Record gradient norm (simplification)
                        gradient_norms[model_name][epoch, i] = np.linalg.norm(dh)

                        # Simplified backprop (just to get a rough idea)
                        dh_next = (
                            dh * 0.5
                        )  # Decay factor to simulate LSTM's ability to preserve gradients

                elif model_name == "gru":
                    dh_next = np.zeros((model.hidden_size, 1))

                    for i in reversed(range(sequence_length)):
                        dh = np.zeros((model.hidden_size, 1))
                        dh += np.dot(model.Wy.T, dy[i].reshape(-1, 1)) + dh_next

                        # Record gradient norm (simplification)
                        gradient_norms[model_name][epoch, i] = np.linalg.norm(dh)

                        # Simplified backprop (just to get a rough idea)
                        dh_next = (
                            dh * 0.6
                        )  # Decay factor to simulate GRU's ability to preserve gradients

    # Train models for a few epochs and record gradients
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train for one epoch
        _ = rnn.train(X, y, epochs=1, print_every=1)
        _ = lstm.train(X, y, epochs=1, print_every=1)
        _ = gru.train(X, y, epochs=1, print_every=1)

        # Record gradients
        record_gradients("rnn", rnn, epoch)
        record_gradients("lstm", lstm, epoch)
        record_gradients("gru", gru, epoch)

    # Plot gradient norms for the last epoch
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.semilogy(gradient_norms["rnn"][epochs - 1], "r", label="RNN")
    plt.semilogy(gradient_norms["lstm"][epochs - 1], "g", label="LSTM")
    plt.semilogy(gradient_norms["gru"][epochs - 1], "b", label="GRU")
    plt.title(f"Gradient Norms at Epoch {epochs}")
    plt.xlabel("Time Step (0 = earliest in sequence)")
    plt.ylabel("Gradient Norm (log scale)")
    plt.legend()
    plt.grid(True)

    # Plot average gradient norms over epochs
    plt.subplot(1, 2, 2)
    plt.plot(np.mean(gradient_norms["rnn"], axis=1), "r", label="RNN")
    plt.plot(np.mean(gradient_norms["lstm"], axis=1), "g", label="LSTM")
    plt.plot(np.mean(gradient_norms["gru"], axis=1), "b", label="GRU")
    plt.title("Average Gradient Norms Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Gradient Norm")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to run the sequence comparison exercises.
    """
    print("Sequence Prediction with LSTM and GRU")
    print("====================================")

    # Compare models on sine wave prediction
    print("\nComparing models on sine wave prediction...")
    sine_wave_results = compare_models_on_sine_wave()

    # Compare models on long-term dependency task
    print("\nComparing models on long-term dependency task...")
    long_term_results = compare_models_on_long_term_dependency()

    # Compare training speed
    print("\nComparing training speed...")
    speed_results = compare_training_speed()

    # Visualize gradient flow
    print("\nVisualizing gradient flow...")
    visualize_gradient_flow(epochs=10)

    print("\nKey takeaways from comparisons:")
    print(
        "1. LSTM and GRU generally outperform standard RNNs, especially for tasks requiring long-term memory."
    )
    print(
        "2. GRU often achieves similar performance to LSTM with less computational complexity."
    )
    print(
        "3. The vanishing gradient problem is evident in standard RNNs but mitigated in LSTM and GRU."
    )
    print(
        "4. For simple sequence tasks (like sine wave prediction), the difference may be small."
    )
    print(
        "5. For tasks requiring memory of events far in the past, LSTM and GRU have a significant advantage."
    )

    print("\nExercise completed!")


if __name__ == "__main__":
    main()
