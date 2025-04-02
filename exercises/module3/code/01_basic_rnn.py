#!/usr/bin/env python3
"""
Basic RNN Implementation - Exercise 1
Implement a Recurrent Neural Network from scratch using NumPy
"""

import numpy as np
import matplotlib.pyplot as plt


class RNN:
    """
    A simple Recurrent Neural Network implemented from scratch using NumPy.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN with the given sizes.

        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
        """
        # Xavier/Glorot initialization for weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output

        # Initialize biases as zeros
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias

        # Store dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Store states for backpropagation
        self.reset_states()

    def reset_states(self):
        """Reset the states stored for backpropagation through time."""
        self.last_inputs = {}
        self.last_hs = {}
        self.last_outputs = {}

        # Initial hidden state is set to zero
        self.h = np.zeros((self.hidden_size, 1))

    def forward(self, x, h_prev=None):
        """
        Forward pass through the RNN for a single time step.

        Args:
            x: Input at current time step, shape (input_size, 1)
            h_prev: Previous hidden state (optional), shape (hidden_size, 1)

        Returns:
            y: Output at current time step
            h: Hidden state at current time step
        """
        if h_prev is None:
            h_prev = self.h

        # Compute hidden state
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)

        # Compute output
        y = np.dot(self.Why, h) + self.by

        # Update hidden state
        self.h = h

        return y, h

    def forward_sequence(self, x_sequence):
        """
        Forward pass through the RNN for a sequence of inputs.

        Args:
            x_sequence: Sequence of inputs, shape (sequence_length, input_size)

        Returns:
            outputs: Sequence of outputs
            hidden_states: Sequence of hidden states
        """
        # Reset states at the beginning of each sequence
        self.reset_states()

        outputs = []
        hidden_states = []

        for t, x in enumerate(x_sequence):
            # Reshape x to be a column vector
            x = x.reshape(-1, 1)

            # Store for backpropagation
            self.last_inputs[t] = x
            self.last_hs[t] = self.h.copy()

            # Forward pass for single time step
            y, h = self.forward(x)

            # Store output and hidden state
            self.last_outputs[t] = y
            outputs.append(y)
            hidden_states.append(h)

        return outputs, hidden_states

    def backward(self, targets, learning_rate=0.01, clip_value=5.0):
        """
        Backpropagation through time to compute gradients and update weights.

        Args:
            targets: Sequence of target outputs
            learning_rate: Learning rate for weight updates
            clip_value: Value to clip gradients to prevent explosion

        Returns:
            loss: Mean squared error loss
        """
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Initialize loss
        loss = 0

        # Initialize gradient of hidden state at final time step
        dh_next = np.zeros_like(self.h)

        # Backpropagate through time
        for t in reversed(range(len(targets))):
            # Current target
            target = targets[t].reshape(-1, 1)

            # Current output
            y = self.last_outputs[t]

            # Compute loss (mean squared error)
            loss += np.sum((y - target) ** 2) / 2

            # Gradient of output
            dy = y - target

            # Gradient of Why
            dWhy += np.dot(dy, self.last_hs[t].T)
            dby += dy

            # Gradient of hidden state
            dh = np.dot(self.Why.T, dy) + dh_next

            # Gradient of tanh
            dh_raw = (1 - self.last_hs[t] ** 2) * dh

            # Gradient of biases
            dbh += dh_raw

            # Gradient of Whh
            dWhh += (
                np.dot(dh_raw, self.last_hs[t - 1].T)
                if t > 0
                else np.dot(dh_raw, np.zeros((self.hidden_size, 1)).T)
            )

            # Gradient of Wxh
            dWxh += np.dot(dh_raw, self.last_inputs[t].T)

            # Gradient of next hidden state
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Average loss
        loss /= len(targets)

        # Clip gradients to prevent explosion
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Update weights
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        return loss

    def train(self, X, y, learning_rate=0.01, epochs=100, clip_value=5.0, verbose=True):
        """
        Train the RNN on sequences of inputs.

        Args:
            X: List of input sequences, each of shape (sequence_length, input_size)
            y: List of target sequences, each of shape (sequence_length, output_size)
            learning_rate: Learning rate for weight updates
            epochs: Number of training epochs
            clip_value: Value to clip gradients to prevent explosion
            verbose: Whether to print progress

        Returns:
            losses: List of losses during training
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            # Loop through each sequence
            for i in range(len(X)):
                # Forward pass
                outputs, _ = self.forward_sequence(X[i])

                # Backward pass
                loss = self.backward(y[i], learning_rate, clip_value)

                epoch_loss += loss

            # Average loss for this epoch
            avg_loss = epoch_loss / len(X)
            losses.append(avg_loss)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, x_sequence):
        """
        Make predictions for a sequence of inputs.

        Args:
            x_sequence: Sequence of inputs, shape (sequence_length, input_size)

        Returns:
            predictions: Sequence of predicted outputs
        """
        outputs, _ = self.forward_sequence(x_sequence)
        return outputs


def generate_simple_patterns():
    """
    Generate simple sequence patterns for training.

    Returns:
        X_train: Training input sequences
        y_train: Training target sequences (shifted by 1)
        X_val: Validation input sequences
        y_val: Validation target sequences
    """
    # 1. Sine Wave
    t = np.linspace(0, 20, 200)
    sine_wave = np.sin(t)

    # 2. Binary Counter (alternating 0, 1 pattern)
    binary_counter = np.array([i % 2 for i in range(100)])

    # 3. Fibonacci-inspired Sequence
    fibonacci_like = np.zeros(50)
    fibonacci_like[0], fibonacci_like[1] = 0.1, 0.1
    for i in range(2, 50):
        fibonacci_like[i] = fibonacci_like[i - 1] + fibonacci_like[i - 2]
    # Normalize to a reasonable range
    fibonacci_like = fibonacci_like / np.max(fibonacci_like)

    # Create sequences for training
    sequence_length = 10

    # Train-validation split
    split_point = int(0.8 * len(sine_wave))

    # Sine wave sequences
    X_sine, y_sine = [], []
    for i in range(len(sine_wave) - sequence_length - 1):
        X_sine.append(sine_wave[i : i + sequence_length])
        y_sine.append(sine_wave[i + 1 : i + sequence_length + 1])

    X_sine_train = X_sine[:split_point]
    y_sine_train = y_sine[:split_point]
    X_sine_val = X_sine[split_point:]
    y_sine_val = y_sine[split_point:]

    # Binary counter sequences
    X_binary, y_binary = [], []
    for i in range(len(binary_counter) - sequence_length - 1):
        X_binary.append(binary_counter[i : i + sequence_length])
        y_binary.append(binary_counter[i + 1 : i + sequence_length + 1])

    split_point = int(0.8 * len(X_binary))
    X_binary_train = X_binary[:split_point]
    y_binary_train = y_binary[:split_point]
    X_binary_val = X_binary[split_point:]
    y_binary_val = y_binary[split_point:]

    # Fibonacci-like sequences
    X_fib, y_fib = [], []
    for i in range(len(fibonacci_like) - sequence_length - 1):
        X_fib.append(fibonacci_like[i : i + sequence_length])
        y_fib.append(fibonacci_like[i + 1 : i + sequence_length + 1])

    split_point = int(0.8 * len(X_fib))
    X_fib_train = X_fib[:split_point]
    y_fib_train = y_fib[:split_point]
    X_fib_val = X_fib[split_point:]
    y_fib_val = y_fib[split_point:]

    # Reshape sequences for RNN input
    # Our RNN expects input of shape (sequence_length, input_size=1)
    X_sine_train = [x.reshape(-1, 1) for x in X_sine_train]
    y_sine_train = [y.reshape(-1, 1) for y in y_sine_train]
    X_sine_val = [x.reshape(-1, 1) for x in X_sine_val]
    y_sine_val = [y.reshape(-1, 1) for y in y_sine_val]

    X_binary_train = [x.reshape(-1, 1) for x in X_binary_train]
    y_binary_train = [y.reshape(-1, 1) for y in y_binary_train]
    X_binary_val = [x.reshape(-1, 1) for x in X_binary_val]
    y_binary_val = [y.reshape(-1, 1) for y in y_binary_val]

    X_fib_train = [x.reshape(-1, 1) for x in X_fib_train]
    y_fib_train = [y.reshape(-1, 1) for y in y_fib_train]
    X_fib_val = [x.reshape(-1, 1) for x in X_fib_val]
    y_fib_val = [y.reshape(-1, 1) for y in y_fib_val]

    return {
        "sine": (X_sine_train, y_sine_train, X_sine_val, y_sine_val, sine_wave),
        "binary": (
            X_binary_train,
            y_binary_train,
            X_binary_val,
            y_binary_val,
            binary_counter,
        ),
        "fibonacci": (X_fib_train, y_fib_train, X_fib_val, y_fib_val, fibonacci_like),
    }


def visualize_hidden_states(rnn, X, sequence_name):
    """
    Visualize the hidden states of the RNN for a given input sequence.

    Args:
        rnn: Trained RNN model
        X: Input sequence
        sequence_name: Name of the sequence for plotting
    """
    # Get a sample sequence
    sample_seq = X[0]

    # Forward pass
    _, hidden_states = rnn.forward_sequence(sample_seq)

    # Convert list of hidden states to numpy array
    hidden_states = np.hstack([h for h in hidden_states])

    # Plot hidden states
    plt.figure(figsize=(10, 6))
    plt.imshow(hidden_states, aspect="auto", cmap="viridis")
    plt.colorbar(label="Activation")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden Unit")
    plt.title(f"Hidden State Evolution for {sequence_name} Sequence")
    plt.tight_layout()
    plt.show()


def train_on_simple_sequence(sequence_type, hidden_size=16, lr=0.01, epochs=100):
    """
    Train an RNN on a simple sequence and evaluate its performance.

    Args:
        sequence_type: Type of sequence ('sine', 'binary', or 'fibonacci')
        hidden_size: Size of the hidden layer
        lr: Learning rate
        epochs: Number of training epochs

    Returns:
        rnn: Trained RNN model
        val_loss: Validation loss
    """
    # Generate sequences
    sequences = generate_simple_patterns()

    if sequence_type not in sequences:
        raise ValueError(
            f"Sequence type '{sequence_type}' not recognized. Choose from: {list(sequences.keys())}"
        )

    X_train, y_train, X_val, y_val, full_sequence = sequences[sequence_type]

    # Create RNN
    input_size = 1  # Each element is a scalar
    output_size = 1  # Predicting a scalar

    rnn = RNN(input_size, hidden_size, output_size)

    # Train RNN
    print(f"\nTraining RNN on {sequence_type} sequence...")
    losses = rnn.train(X_train, y_train, learning_rate=lr, epochs=epochs)

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title(f"Training Loss for {sequence_type} Sequence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate on validation set
    val_loss = 0
    predictions = []

    for i in range(len(X_val)):
        # Get predictions
        outputs = rnn.predict(X_val[i])
        predictions.append(outputs)

        # Compute validation loss
        mse = np.mean([(outputs[j] - y_val[i][j]) ** 2 for j in range(len(outputs))])
        val_loss += mse

    val_loss /= len(X_val)
    print(f"Validation Loss: {val_loss:.6f}")

    # Plot predictions for a few validation sequences
    plt.figure(figsize=(12, 6))

    for i in range(min(3, len(X_val))):
        plt.subplot(3, 1, i + 1)

        # Original input
        input_vals = [x[0][0] for x in X_val[i]]
        plt.plot(input_vals, "b.-", label="Input")

        # Target
        target_vals = [y[0][0] for y in y_val[i]]
        plt.plot(target_vals, "g.-", label="Target")

        # Prediction
        pred_vals = [p[0][0] for p in predictions[i]]
        plt.plot(pred_vals, "r.-", label="Prediction")

        plt.title(f"{sequence_type} Sequence Prediction (Example {i+1})")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Visualize hidden states
    visualize_hidden_states(rnn, X_val, sequence_type)

    return rnn, val_loss


def predict_longer_sequence(rnn, seed_sequence, steps_ahead=30, sequence_type=None):
    """
    Use the trained RNN to predict future values beyond the training sequence.

    Args:
        rnn: Trained RNN model
        seed_sequence: Initial sequence to start the prediction
        steps_ahead: Number of steps to predict ahead
        sequence_type: Type of sequence for comparison (if available)
    """
    # Make a copy of the seed sequence
    current_sequence = seed_sequence.copy()

    # Initialize list to store predictions
    predictions = []

    # Generate predictions one step at a time
    for _ in range(steps_ahead):
        # Get the latest sequence window
        latest_window = (
            current_sequence[-10:] if len(current_sequence) >= 10 else current_sequence
        )

        # Reshape for RNN input
        rnn_input = np.array(latest_window).reshape(-1, 1)

        # Make prediction for the next step
        _, h = rnn.forward_sequence(rnn_input)
        next_val = rnn.forward(np.array(latest_window[-1]).reshape(-1, 1), h[-1])[0]

        # Add prediction to the sequence
        predictions.append(next_val[0][0])
        current_sequence.append(next_val[0][0])

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot original seed sequence
    plt.plot(range(len(seed_sequence)), seed_sequence, "b.-", label="Seed Sequence")

    # Plot predictions
    plt.plot(
        range(len(seed_sequence), len(seed_sequence) + steps_ahead),
        predictions,
        "r.-",
        label="Predictions",
    )

    # Plot ground truth if available
    if sequence_type:
        sequences = generate_simple_patterns()
        full_sequence = sequences[sequence_type][4]
        if len(full_sequence) >= len(seed_sequence) + steps_ahead:
            ground_truth = full_sequence[
                len(seed_sequence) : len(seed_sequence) + steps_ahead
            ]
            plt.plot(
                range(len(seed_sequence), len(seed_sequence) + steps_ahead),
                ground_truth,
                "g.-",
                label="Ground Truth",
            )

    plt.title("RNN Long-Term Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demonstrate_memory_capacity():
    """
    Demonstrate how RNN memory capacity is affected by sequence length and hidden size.
    """
    # Generate a delayed XOR problem
    # For inputs [a, b, c], the target is a XOR c with delay in between
    # Shows if RNN can remember information over time

    # Generate data
    np.random.seed(42)
    sequence_length = 5
    num_sequences = 100
    delay = 3  # Number of time steps between relevant inputs

    X = []
    y = []

    for _ in range(num_sequences):
        # Random binary inputs
        seq = np.random.randint(0, 2, (sequence_length, 1))
        target = np.zeros_like(seq)

        # For each element after the delay, the target is the XOR of the current value
        # and the value 'delay' steps back
        for i in range(delay, sequence_length):
            target[i, 0] = seq[i, 0] ^ seq[i - delay, 0]

        X.append(seq)
        y.append(target)

    # Split data
    split = int(0.8 * num_sequences)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Test different hidden sizes
    hidden_sizes = [4, 16, 32]
    results = []

    for hidden_size in hidden_sizes:
        print(f"\nTraining RNN with hidden size {hidden_size} on delayed XOR task...")

        # Create RNN
        rnn = RNN(input_size=1, hidden_size=hidden_size, output_size=1)

        # Train RNN
        losses = rnn.train(X_train, y_train, learning_rate=0.01, epochs=100)

        # Evaluate on validation set
        correct = 0
        total = 0

        for i in range(len(X_val)):
            outputs = rnn.predict(X_val[i])

            # Count correct predictions (threshold at 0.5)
            for j in range(delay, len(outputs)):
                pred = 1 if outputs[j][0][0] > 0.5 else 0
                target = y_val[i][j][0]
                correct += pred == target
                total += 1

        accuracy = correct / total
        results.append((hidden_size, accuracy))
        print(f"Hidden Size: {hidden_size}, Validation Accuracy: {accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    hidden_sizes, accuracies = zip(*results)
    plt.bar(range(len(hidden_sizes)), accuracies, tick_label=hidden_sizes)
    plt.xlabel("Hidden Size")
    plt.ylabel("Accuracy")
    plt.title("Effect of Hidden Size on RNN Memory Capacity")
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")
    plt.tight_layout()
    plt.show()

    # Test different sequence lengths with fixed hidden size
    hidden_size = 16
    delays = [1, 3, 5, 7]
    results = []

    for delay in delays:
        print(f"\nTesting RNN memory with delay {delay}...")

        # Generate data with the current delay
        X = []
        y = []

        for _ in range(num_sequences):
            seq = np.random.randint(0, 2, (sequence_length, 1))
            target = np.zeros_like(seq)

            # Only calculate targets for positions after the delay
            for i in range(delay, sequence_length):
                target[i, 0] = seq[i, 0] ^ seq[i - delay, 0]

            X.append(seq)
            y.append(target)

        # Split data
        split = int(0.8 * num_sequences)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # Create RNN
        rnn = RNN(input_size=1, hidden_size=hidden_size, output_size=1)

        # Train RNN
        losses = rnn.train(
            X_train, y_train, learning_rate=0.01, epochs=100, verbose=False
        )

        # Evaluate on validation set
        correct = 0
        total = 0

        for i in range(len(X_val)):
            outputs = rnn.predict(X_val[i])

            # Count correct predictions (threshold at 0.5)
            for j in range(delay, len(outputs)):
                pred = 1 if outputs[j][0][0] > 0.5 else 0
                target = y_val[i][j][0]
                correct += pred == target
                total += 1

        accuracy = correct / total
        results.append((delay, accuracy))
        print(f"Delay: {delay}, Validation Accuracy: {accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    delays, accuracies = zip(*results)
    plt.plot(delays, accuracies, "o-")
    plt.xlabel("Delay (Time Steps)")
    plt.ylabel("Accuracy")
    plt.title("Effect of Sequence Delay on RNN Memory Capacity")
    plt.grid(True)
    plt.xticks(delays)
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(delays[i], v + 0.02, f"{v:.4f}", ha="center")
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the RNN experiments."""
    print("===== Basic RNN Implementation Exercise =====")

    # 1. Train on sine wave
    print("\n1. Training RNN on Sine Wave")
    sine_rnn, _ = train_on_simple_sequence("sine", hidden_size=16, lr=0.01, epochs=200)

    # Get a sample seed sequence
    sequences = generate_simple_patterns()
    seed_sequence = [x[0][0] for x in sequences["sine"][2][0]]

    # Predict future values
    predict_longer_sequence(
        sine_rnn, seed_sequence, steps_ahead=30, sequence_type="sine"
    )

    # 2. Train on binary sequence
    print("\n2. Training RNN on Binary Sequence")
    binary_rnn, _ = train_on_simple_sequence(
        "binary", hidden_size=8, lr=0.01, epochs=200
    )

    # 3. Train on Fibonacci-like sequence
    print("\n3. Training RNN on Fibonacci-like Sequence")
    fib_rnn, _ = train_on_simple_sequence(
        "fibonacci", hidden_size=16, lr=0.01, epochs=200
    )

    # 4. Demonstrate memory capacity
    print("\n4. Demonstrating RNN Memory Capacity")
    demonstrate_memory_capacity()

    print("\nThis exercise demonstrated:")
    print("- How to implement a basic RNN from scratch")
    print("- How RNNs can learn simple sequence patterns")
    print("- How hidden states evolve over time")
    print("- How the memory capacity of an RNN depends on its architecture")


if __name__ == "__main__":
    main()
