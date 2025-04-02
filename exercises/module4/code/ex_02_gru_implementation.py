"""
GRU Implementation from Scratch

This exercise implements a Gated Recurrent Unit (GRU) network from scratch
using NumPy to understand its architecture and advantages compared to LSTM.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


class GRUCell:
    """
    A single Gated Recurrent Unit (GRU) cell implementation.

    The GRU cell contains:
    - Update gate: controls how much of the previous hidden state to keep
    - Reset gate: controls how much of the previous hidden state to reset
    - New hidden state: computes the new hidden state candidate
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize a GRU cell with random weights.

        Args:
            input_size: Size of the input vector
            hidden_size: Size of the hidden state vector
        """
        # Weight initialization with Xavier/Glorot initialization for better gradient flow
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Update gate weights and biases
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bz = np.zeros((hidden_size, 1))

        # Reset gate weights and biases
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.br = np.zeros((hidden_size, 1))

        # New hidden state weights and biases
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))

        # Memory for storing activations for backpropagation
        self.cache = {}

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for a single time step.

        Args:
            x_t: Input at the current time step (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)

        Returns:
            h_t: Current hidden state
        """
        # Reshape inputs for matrix multiplication if needed
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)

        # Concatenate input and previous hidden state
        combined = np.vstack((h_prev, x_t))

        # Update gate
        z_t = self._sigmoid(np.dot(self.Wz, combined) + self.bz)

        # Reset gate
        r_t = self._sigmoid(np.dot(self.Wr, combined) + self.br)

        # Reset gate applied to previous hidden state
        reset_h = r_t * h_prev

        # Calculate candidate hidden state
        combined_reset = np.vstack((reset_h, x_t))
        h_candidate = np.tanh(np.dot(self.Wh, combined_reset) + self.bh)

        # Calculate new hidden state using update gate
        h_t = (1 - z_t) * h_prev + z_t * h_candidate

        # Store values for backpropagation
        self.cache = {
            "x_t": x_t,
            "h_prev": h_prev,
            "combined": combined,
            "z_t": z_t,
            "r_t": r_t,
            "reset_h": reset_h,
            "combined_reset": combined_reset,
            "h_candidate": h_candidate,
            "h_t": h_t,
        }

        return h_t

    def backward(self, dh_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for a single time step.

        Args:
            dh_t: Gradient of the loss with respect to the hidden state

        Returns:
            dx_t: Gradient with respect to the input
            dh_prev: Gradient with respect to the previous hidden state
        """
        # Reshape gradients if needed
        if dh_t.ndim == 1:
            dh_t = dh_t.reshape(-1, 1)

        # Retrieve stored values from forward pass
        x_t = self.cache["x_t"]
        h_prev = self.cache["h_prev"]
        combined = self.cache["combined"]
        z_t = self.cache["z_t"]
        r_t = self.cache["r_t"]
        reset_h = self.cache["reset_h"]
        combined_reset = self.cache["combined_reset"]
        h_candidate = self.cache["h_candidate"]

        # Gradient through the hidden state equation
        dh_prev_partial = dh_t * (1 - z_t)
        dh_candidate = dh_t * z_t
        dz_t = dh_t * (h_candidate - h_prev)

        # Gradient through the candidate hidden state
        dcombined_reset = np.dot(self.Wh.T, dh_candidate * (1 - h_candidate**2))

        # Split gradients between reset hidden state and input
        hidden_size = h_prev.shape[0]
        dreset_h = dcombined_reset[:hidden_size]
        dx_t_1 = dcombined_reset[hidden_size:]

        # Gradient through the reset gate
        dh_prev_reset = dreset_h * r_t
        dr_t = dreset_h * h_prev

        # Gradient through the update and reset gates
        dz_combined = dz_t * z_t * (1 - z_t)
        dr_combined = dr_t * r_t * (1 - r_t)

        # Gradient with respect to the weight matrices
        dWz = np.dot(dz_combined, combined.T)
        dWr = np.dot(dr_combined, combined.T)
        dWh = np.dot(dh_candidate * (1 - h_candidate**2), combined_reset.T)

        # Gradient with respect to the biases
        dbz = np.sum(dz_combined, axis=1, keepdims=True)
        dbr = np.sum(dr_combined, axis=1, keepdims=True)
        dbh = np.sum(dh_candidate * (1 - h_candidate**2), axis=1, keepdims=True)

        # Gradient with respect to the combined input for update and reset gates
        dcombined = np.dot(self.Wz.T, dz_combined) + np.dot(self.Wr.T, dr_combined)

        # Split gradients for previous hidden state and input
        dh_prev_gates = dcombined[:hidden_size]
        dx_t_2 = dcombined[hidden_size:]

        # Combine gradients for the previous hidden state
        dh_prev = dh_prev_partial + dh_prev_reset + dh_prev_gates

        # Combine gradients for the input
        dx_t = dx_t_1 + dx_t_2

        # Store gradients for parameter updates
        self.dWz, self.dbz = dWz, dbz
        self.dWr, self.dbr = dWr, dbr
        self.dWh, self.dbh = dWh, dbh

        return dx_t, dh_prev

    def update_params(self, learning_rate: float = 0.01) -> None:
        """
        Update the parameters of the GRU cell using the calculated gradients.

        Args:
            learning_rate: Learning rate for gradient descent
        """
        # Update weights
        self.Wz -= learning_rate * self.dWz
        self.Wr -= learning_rate * self.dWr
        self.Wh -= learning_rate * self.dWh

        # Update biases
        self.bz -= learning_rate * self.dbz
        self.br -= learning_rate * self.dbr
        self.bh -= learning_rate * self.dbh

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            x: Input to the sigmoid function

        Returns:
            Sigmoid of the input
        """
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))


class GRU:
    """
    GRU network made up of GRU cells.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a GRU network.

        Args:
            input_size: Size of the input vector
            hidden_size: Size of the hidden state vector
            output_size: Size of the output vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize GRU cell
        self.gru_cell = GRUCell(input_size, hidden_size)

        # Output layer
        scale = np.sqrt(2.0 / (hidden_size + output_size))
        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

        # Memory for storing states and activations
        self.h_states = {}
        self.outputs = {}

    def forward(self, x_sequence: np.ndarray) -> np.ndarray:
        """
        Forward pass through the GRU network for the entire sequence.

        Args:
            x_sequence: Input sequence of shape (sequence_length, input_size)

        Returns:
            y_sequence: Output sequence of shape (sequence_length, output_size)
        """
        sequence_length = len(x_sequence)

        # Initialize hidden state
        h_t = np.zeros((self.hidden_size, 1))

        # Process each time step
        y_sequence = np.zeros((sequence_length, self.output_size))

        for t in range(sequence_length):
            # Get input at current time step
            x_t = x_sequence[t].reshape(-1, 1)

            # Forward pass through GRU cell
            h_t = self.gru_cell.forward(x_t, h_t)

            # Store states for backpropagation
            self.h_states[t] = h_t

            # Compute output
            y_t = np.dot(self.Wy, h_t) + self.by

            # Store output
            self.outputs[t] = y_t
            y_sequence[t] = y_t.flatten()

        return y_sequence

    def backward(self, dy_sequence: np.ndarray, learning_rate: float = 0.01) -> None:
        """
        Backward pass through the GRU network for the entire sequence.

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

            # Backward pass through GRU cell
            _, dh_next = self.gru_cell.backward(dh_t)

            # Update GRU cell parameters
            self.gru_cell.update_params(learning_rate)

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
        Train the GRU network on sequence data.

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
    model: GRU, X_test: np.ndarray, y_test: np.ndarray, sequence_length: int = 100
) -> None:
    """
    Visualize predictions from the GRU model.

    Args:
        model: Trained GRU model
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
    plt.title("GRU Sequence Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()


def demonstrate_sequence_copy_task() -> None:
    """
    Demonstrate the ability of GRU to remember sequences by implementing a sequence copy task.
    The model must remember a sequence of random numbers and reproduce it after a delay.
    """
    # Task parameters
    seq_length = 10
    delay_length = 20
    num_sequences = 100

    # Generate data: random sequences with a delay, then the sequence should be repeated
    X = np.zeros((num_sequences, seq_length + delay_length + seq_length, 1))
    y = np.zeros_like(X)

    for i in range(num_sequences):
        # Generate random sequence
        random_seq = np.random.rand(seq_length, 1)

        # Input: random sequence followed by zeros (delay)
        X[i, :seq_length, 0] = random_seq.flatten()

        # Target: zeros during sequence and delay, then the sequence
        y[i, seq_length + delay_length :, 0] = random_seq.flatten()

    # Split into train and test
    split = int(0.8 * num_sequences)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train GRU model
    print("Training GRU on sequence copy task...")
    gru_model = GRU(input_size=1, hidden_size=20, output_size=1)
    losses = gru_model.train(
        X_train, y_train, epochs=200, learning_rate=0.01, print_every=20
    )

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("GRU Training Loss on Sequence Copy Task")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    # Visualize a test example
    sample_idx = np.random.randint(0, len(X_test))
    sample_X = X_test[sample_idx]
    sample_y = y_test[sample_idx]
    predictions = gru_model.forward(sample_X)

    # Plot the input, target, and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(sample_X, "b", label="Input Sequence")
    plt.plot(sample_y, "g", label="Target Sequence")
    plt.plot(predictions, "r--", label="GRU Predictions")

    # Add vertical lines to mark the sequence and delay periods
    plt.axvline(
        x=seq_length - 1,
        color="k",
        linestyle="--",
        alpha=0.5,
        label="End of Input Sequence",
    )
    plt.axvline(
        x=seq_length + delay_length - 1,
        color="k",
        linestyle=":",
        alpha=0.5,
        label="End of Delay Period",
    )

    plt.legend()
    plt.title("GRU Sequence Copy Task")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def compare_gru_hidden_sizes() -> None:
    """
    Compare GRU models with different hidden sizes on a sine wave prediction task.
    """
    # Generate data
    X, y = generate_sine_wave(samples=1000, period=20)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for GRU input (sequence_length, input_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    # Try different hidden sizes
    hidden_sizes = [5, 10, 20]
    losses_dict = {}

    for hidden_size in hidden_sizes:
        print(f"\nTraining GRU with hidden size {hidden_size}")
        gru_model = GRU(input_size=1, hidden_size=hidden_size, output_size=1)
        losses = gru_model.train(
            X_train, y_train, epochs=50, learning_rate=0.01, print_every=10
        )
        losses_dict[hidden_size] = losses

        # Evaluate on test set
        test_loss = 0
        for i in range(len(X_test)):
            y_pred = gru_model.forward(X_test[i])
            test_loss += np.mean((y_pred - y_test[i].reshape(-1)) ** 2)
        test_loss /= len(X_test)
        print(f"Test loss for hidden size {hidden_size}: {test_loss:.4f}")

    # Plot losses for different hidden sizes
    plt.figure(figsize=(10, 6))
    for hidden_size, losses in losses_dict.items():
        plt.plot(losses, label=f"Hidden Size: {hidden_size}")
    plt.title("GRU Training Loss for Different Hidden Sizes")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def explain_gru_mechanics() -> None:
    """
    Create a visualization to explain the mechanics of the GRU gates.
    """
    # Create a sample input sequence
    sequence_length = 100
    x = np.linspace(0, 4 * np.pi, sequence_length)
    input_sequence = np.sin(x).reshape(sequence_length, 1)

    # Create a small GRU with a single hidden unit for visualization
    gru_cell = GRUCell(input_size=1, hidden_size=1)

    # Initialize hidden state
    h_t = np.zeros((1, 1))

    # Track gate values
    update_gates = np.zeros(sequence_length)
    reset_gates = np.zeros(sequence_length)
    candidates = np.zeros(sequence_length)
    hidden_states = np.zeros(sequence_length)

    # Process sequence
    for t in range(sequence_length):
        x_t = input_sequence[t]
        h_t = gru_cell.forward(x_t, h_t)

        # Record gate values
        update_gates[t] = gru_cell.cache["z_t"]
        reset_gates[t] = gru_cell.cache["r_t"]
        candidates[t] = gru_cell.cache["h_candidate"]
        hidden_states[t] = h_t

    # Plot the gates and their effects
    plt.figure(figsize=(12, 8))

    # Input and hidden state
    plt.subplot(4, 1, 1)
    plt.plot(input_sequence, "b", label="Input")
    plt.plot(hidden_states, "r", label="Hidden State")
    plt.legend()
    plt.title("GRU Mechanics Visualization")
    plt.ylabel("Value")
    plt.grid(True)

    # Update gate
    plt.subplot(4, 1, 2)
    plt.plot(update_gates, "g", label="Update Gate")
    plt.axhline(y=0.5, color="k", linestyle="--", alpha=0.3)
    plt.legend()
    plt.ylabel("Gate Value")
    plt.text(
        sequence_length / 2, 0.7, "Controls how much new info to add vs. keep old info"
    )
    plt.grid(True)

    # Reset gate
    plt.subplot(4, 1, 3)
    plt.plot(reset_gates, "m", label="Reset Gate")
    plt.axhline(y=0.5, color="k", linestyle="--", alpha=0.3)
    plt.legend()
    plt.ylabel("Gate Value")
    plt.text(sequence_length / 2, 0.7, "Controls how much past info to forget")
    plt.grid(True)

    # Candidate hidden state
    plt.subplot(4, 1, 4)
    plt.plot(candidates, "c", label="Candidate Hidden State")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.text(
        sequence_length / 2, 0.5, "New information to potentially add to hidden state"
    )
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Create a diagram showing how the hidden state is updated
    plt.figure(figsize=(12, 6))

    # Plot a segment of the sequence for clarity
    segment_start = 40
    segment_end = 60
    t_segment = range(segment_start, segment_end)

    plt.subplot(2, 1, 1)
    plt.plot(t_segment, input_sequence[segment_start:segment_end], "b", label="Input")
    plt.plot(
        t_segment, hidden_states[segment_start:segment_end], "r", label="Hidden State"
    )
    plt.legend()
    plt.title("GRU Hidden State Update Mechanics")
    plt.ylabel("Value")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(
        t_segment, update_gates[segment_start:segment_end], "g", label="Update Gate"
    )
    plt.plot(t_segment, candidates[segment_start:segment_end], "c", label="Candidate")
    plt.plot(
        t_segment,
        hidden_states[segment_start - 1 : segment_end - 1],
        "r--",
        alpha=0.5,
        label="Previous Hidden",
    )
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.text(segment_start + 5, 0.5, "h_t = (1-z_t)*h_{t-1} + z_t*h_candidate")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to run the GRU exercises.
    """
    print("GRU Implementation from Scratch")
    print("==============================")

    # Generate sine wave data
    X, y = generate_sine_wave(samples=500, period=20)
    print(f"Generated {len(X)} sequences for training")

    # Reshape data for GRU input (sequence_length, input_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train GRU model
    print("\nTraining GRU model...")
    gru_model = GRU(input_size=1, hidden_size=10, output_size=1)
    losses = gru_model.train(
        X_train, y_train, epochs=100, learning_rate=0.01, print_every=10
    )

    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("GRU Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(gru_model, X_test, y_test)

    # Demonstrate sequence copy task
    print("\nDemonstrating GRU sequence copy task...")
    demonstrate_sequence_copy_task()

    # Compare different GRU architectures
    print("\nComparing different GRU architectures...")
    compare_gru_hidden_sizes()

    # Explain GRU mechanics
    print("\nExplaining GRU mechanics...")
    explain_gru_mechanics()

    print("\nExercise completed!")


if __name__ == "__main__":
    main()
