"""
LSTM Implementation from Scratch

This exercise implements a Long Short-Term Memory (LSTM) network from scratch
using NumPy to gain a deep understanding of its internal mechanisms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


class LSTMCell:
    """
    A single LSTM cell implementation.

    The LSTM cell contains:
    - Forget gate: controls what to forget from the cell state
    - Input gate: controls what new information to add to the cell state
    - Cell state: the memory of the LSTM
    - Output gate: controls what parts of the cell state to output
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize an LSTM cell with random weights.

        Args:
            input_size: Size of the input vector
            hidden_size: Size of the hidden state vector
        """
        # Weight initialization with Xavier/Glorot initialization for better gradient flow
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Forget gate weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))

        # Input gate weights and biases
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))

        # Cell state candidate weights and biases
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))

        # Output gate weights and biases
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))

        # Memory for storing activations for backpropagation
        self.cache = {}

    def forward(
        self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for a single time step.

        Args:
            x_t: Input at the current time step (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)

        Returns:
            h_t: Current hidden state
            c_t: Current cell state
        """
        # Reshape inputs for matrix multiplication if needed
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)

        # Concatenate input and previous hidden state
        combined = np.vstack((h_prev, x_t))

        # Forget gate
        f_gate = self._sigmoid(np.dot(self.Wf, combined) + self.bf)

        # Input gate
        i_gate = self._sigmoid(np.dot(self.Wi, combined) + self.bi)

        # Cell candidate
        c_candidate = np.tanh(np.dot(self.Wc, combined) + self.bc)

        # Output gate
        o_gate = self._sigmoid(np.dot(self.Wo, combined) + self.bo)

        # Update cell state
        c_t = f_gate * c_prev + i_gate * c_candidate

        # Compute hidden state
        h_t = o_gate * np.tanh(c_t)

        # Store values for backpropagation
        self.cache = {
            "x_t": x_t,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "combined": combined,
            "f_gate": f_gate,
            "i_gate": i_gate,
            "c_candidate": c_candidate,
            "o_gate": o_gate,
            "c_t": c_t,
            "h_t": h_t,
        }

        return h_t, c_t

    def backward(
        self, dh_t: np.ndarray, dc_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for a single time step.

        Args:
            dh_t: Gradient of the loss with respect to the hidden state
            dc_t: Gradient of the loss with respect to the cell state

        Returns:
            dx_t: Gradient with respect to the input
            dh_prev: Gradient with respect to the previous hidden state
            dc_prev: Gradient with respect to the previous cell state
        """
        # Reshape gradients if needed
        if dh_t.ndim == 1:
            dh_t = dh_t.reshape(-1, 1)
        if dc_t.ndim == 1:
            dc_t = dc_t.reshape(-1, 1)

        # Retrieve stored values from forward pass
        x_t = self.cache["x_t"]
        h_prev = self.cache["h_prev"]
        c_prev = self.cache["c_prev"]
        combined = self.cache["combined"]
        f_gate = self.cache["f_gate"]
        i_gate = self.cache["i_gate"]
        c_candidate = self.cache["c_candidate"]
        o_gate = self.cache["o_gate"]
        c_t = self.cache["c_t"]

        # Gradient through the hidden state
        do_gate = dh_t * np.tanh(c_t)
        dc_t += dh_t * o_gate * (1 - np.tanh(c_t) ** 2)

        # Gradient through the cell state
        dc_prev = dc_t * f_gate
        df_gate = dc_t * c_prev
        di_gate = dc_t * c_candidate
        dc_candidate = dc_t * i_gate

        # Gradient through the gates
        do_combined = do_gate * o_gate * (1 - o_gate)
        df_combined = df_gate * f_gate * (1 - f_gate)
        di_combined = di_gate * i_gate * (1 - i_gate)
        dc_combined = dc_candidate * (1 - c_candidate**2)

        # Gradient with respect to the weight matrices
        dWo = np.dot(do_combined, combined.T)
        dWf = np.dot(df_combined, combined.T)
        dWi = np.dot(di_combined, combined.T)
        dWc = np.dot(dc_combined, combined.T)

        # Gradient with respect to the biases
        dbo = np.sum(do_combined, axis=1, keepdims=True)
        dbf = np.sum(df_combined, axis=1, keepdims=True)
        dbi = np.sum(di_combined, axis=1, keepdims=True)
        dbc = np.sum(dc_combined, axis=1, keepdims=True)

        # Gradient with respect to the combined input
        dcombined = (
            np.dot(self.Wo.T, do_combined)
            + np.dot(self.Wf.T, df_combined)
            + np.dot(self.Wi.T, di_combined)
            + np.dot(self.Wc.T, dc_combined)
        )

        # Split the gradient between previous hidden state and input
        hidden_size = h_prev.shape[0]
        dh_prev = dcombined[:hidden_size]
        dx_t = dcombined[hidden_size:]

        # Store gradients for parameter updates
        self.dWo, self.dbo = dWo, dbo
        self.dWf, self.dbf = dWf, dbf
        self.dWi, self.dbi = dWi, dbi
        self.dWc, self.dbc = dWc, dbc

        return dx_t, dh_prev, dc_prev

    def update_params(self, learning_rate: float = 0.01) -> None:
        """
        Update the parameters of the LSTM cell using the calculated gradients.

        Args:
            learning_rate: Learning rate for gradient descent
        """
        # Update weights
        self.Wo -= learning_rate * self.dWo
        self.Wf -= learning_rate * self.dWf
        self.Wi -= learning_rate * self.dWi
        self.Wc -= learning_rate * self.dWc

        # Update biases
        self.bo -= learning_rate * self.dbo
        self.bf -= learning_rate * self.dbf
        self.bi -= learning_rate * self.dbi
        self.bc -= learning_rate * self.dbc

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            x: Input to the sigmoid function

        Returns:
            Sigmoid of the input
        """
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))


class LSTM:
    """
    LSTM network made up of LSTM cells.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize an LSTM network.

        Args:
            input_size: Size of the input vector
            hidden_size: Size of the hidden state vector
            output_size: Size of the output vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize LSTM cell
        self.lstm_cell = LSTMCell(input_size, hidden_size)

        # Output layer
        scale = np.sqrt(2.0 / (hidden_size + output_size))
        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

        # Memory for storing states and activations
        self.h_states = {}
        self.c_states = {}
        self.outputs = {}

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """
        Forward pass through the LSTM network for the entire sequence.

        Args:
            x_sequence: Input sequence of shape (sequence_length, input_size)

        Returns:
            y_sequence: Output sequence of shape (sequence_length, output_size)
        """
        sequence_length = len(x_sequence)

        # Initialize hidden state and cell state
        h_t = np.zeros((self.hidden_size, 1))
        c_t = np.zeros((self.hidden_size, 1))

        # Process each time step
        y_sequence = np.zeros((sequence_length, self.output_size))

        for t in range(sequence_length):
            # Get input at current time step
            x_t = x_sequence[t].reshape(-1, 1)

            # Forward pass through LSTM cell
            h_t, c_t = self.lstm_cell.forward(x_t, h_t, c_t)

            # Store states for backpropagation
            self.h_states[t] = h_t
            self.c_states[t] = c_t

            # Compute output
            y_t = np.dot(self.Wy, h_t) + self.by

            # Store output
            self.outputs[t] = y_t
            y_sequence[t] = y_t.flatten()

        return y_sequence

    def backward(self, dy_sequence: np.ndarray, learning_rate: float = 0.01) -> None:
        """
        Backward pass through the LSTM network for the entire sequence.

        Args:
            dy_sequence: Gradient of the loss with respect to the outputs
            learning_rate: Learning rate for gradient descent
        """
        sequence_length = len(dy_sequence)

        # Initialize gradients
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        # Initialize gradients for the last time step
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        # Process gradients backward in time
        for t in reversed(range(sequence_length)):
            # Get gradient for current time step
            dy_t = dy_sequence[t].reshape(-1, 1)

            # Gradient for output layer
            dh_t = np.dot(self.Wy.T, dy_t)
            dWy += np.dot(dy_t, self.h_states[t].T)
            dby += dy_t

            # Add gradient from next time step (if not last in sequence)
            dh_t += dh_next
            dc_t = dc_next

            # Backward pass through LSTM cell
            dx_t, dh_next, dc_next = self.lstm_cell.backward(dh_t, dc_t)

            # Update LSTM cell parameters
            self.lstm_cell.update_params(learning_rate)

        # Update output layer parameters
        self.Wy -= learning_rate * dWy
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
        Train the LSTM network on sequence data.

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


def visualize_predictions(
    model: LSTM, X_test: np.ndarray, y_test: np.ndarray, sequence_length: int = 100
) -> None:
    """
    Visualize predictions from the LSTM model.

    Args:
        model: Trained LSTM model
        X_test: Test input data
        y_test: Test target data
        sequence_length: Length of the sequence to predict
    """
    # Generate a longer sequence for visualization
    idx = np.random.randint(0, len(X_test))
    X_sample = X_test[idx]
    y_sample = y_test[idx]

    # Make prediction
    y_pred = model.forward(X_sample)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(X_sample)), X_sample, "b", label="Input")
    plt.plot(
        range(len(X_sample), len(X_sample) + len(y_sample)),
        y_sample,
        "g",
        label="Target",
    )
    plt.plot(
        range(len(X_sample), len(X_sample) + len(y_pred)),
        y_pred,
        "r--",
        label="Prediction",
    )
    plt.legend()
    plt.title("LSTM Sequence Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()


def demonstrate_memory_capabilities(sequence_length: int = 100) -> None:
    """
    Demonstrate the memory capabilities of LSTM by creating a task
    that requires remembering information from the beginning of the sequence.

    Args:
        sequence_length: Length of the sequences to generate
    """
    # Generate data where the output depends on the first element of the sequence
    num_sequences = 100
    X = np.random.randn(num_sequences, sequence_length, 1)

    # Create target where the output is the first element repeated
    y = np.zeros_like(X)
    for i in range(num_sequences):
        y[i, :, 0] = X[i, 0, 0]

    # Split into train and test
    split = int(0.8 * num_sequences)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train LSTM model
    lstm_model = LSTM(input_size=1, hidden_size=10, output_size=1)
    losses = lstm_model.train(X_train, y_train, epochs=100, learning_rate=0.01)

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")  # Log scale to better see the decreasing loss
    plt.grid(True)
    plt.show()

    # Make predictions on a test sequence
    sample_idx = np.random.randint(0, len(X_test))
    sample_X = X_test[sample_idx]
    sample_y = y_test[sample_idx]
    predictions = lstm_model.forward(sample_X)

    # Visualize the first value and the predictions
    plt.figure(figsize=(12, 6))
    plt.axhline(
        y=sample_X[0, 0],
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"First Value: {sample_X[0, 0]:.2f}",
    )
    plt.plot(predictions, label="LSTM Predictions")
    plt.plot(sample_y, color="g", alpha=0.5, label="Target")
    plt.legend()
    plt.title("LSTM Memory Test: Remembering First Value in Sequence")
    plt.xlabel("Sequence Position")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def compare_lstm_architectures() -> None:
    """
    Compare different LSTM architectures with varying hidden sizes on a sequence prediction task.
    """
    # Generate data
    X, y = generate_sine_wave(samples=1000, period=20)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM input (sequence_length, input_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    # Try different hidden sizes
    hidden_sizes = [5, 10, 20]
    losses_dict = {}

    for hidden_size in hidden_sizes:
        print(f"\nTraining LSTM with hidden size {hidden_size}")
        lstm_model = LSTM(input_size=1, hidden_size=hidden_size, output_size=1)
        losses = lstm_model.train(
            X_train, y_train, epochs=50, learning_rate=0.01, print_every=10
        )
        losses_dict[hidden_size] = losses

        # Evaluate on test set
        test_loss = 0
        for i in range(len(X_test)):
            y_pred = lstm_model.forward(X_test[i])
            test_loss += np.mean((y_pred - y_test[i].reshape(-1)) ** 2)
        test_loss /= len(X_test)
        print(f"Test loss for hidden size {hidden_size}: {test_loss:.4f}")

    # Plot losses for different hidden sizes
    plt.figure(figsize=(10, 6))
    for hidden_size, losses in losses_dict.items():
        plt.plot(losses, label=f"Hidden Size: {hidden_size}")
    plt.title("LSTM Training Loss for Different Hidden Sizes")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """
    Main function to run the LSTM exercises.
    """
    print("LSTM Implementation from Scratch")
    print("===============================")

    # Generate sine wave data
    X, y = generate_sine_wave(samples=500, period=20)
    print(f"Generated {len(X)} sequences for training")

    # Reshape data for LSTM input (sequence_length, input_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train LSTM model
    print("\nTraining LSTM model...")
    lstm_model = LSTM(input_size=1, hidden_size=10, output_size=1)
    losses = lstm_model.train(
        X_train, y_train, epochs=100, learning_rate=0.01, print_every=10
    )

    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(lstm_model, X_test, y_test)

    # Demonstrate memory capabilities
    print("\nDemonstrating LSTM memory capabilities...")
    demonstrate_memory_capabilities()

    # Compare different LSTM architectures
    print("\nComparing different LSTM architectures...")
    compare_lstm_architectures()

    print("\nExercise completed!")


if __name__ == "__main__":
    main()
