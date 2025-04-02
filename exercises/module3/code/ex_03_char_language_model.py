#!/usr/bin/env python3
"""
Character-Level Language Model - Exercise 3
Build a character-level language model using RNNs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from io import StringIO
import time
import string
import random


class CharRNN:
    """
    Character-level Recurrent Neural Network for language modeling.
    """

    def __init__(self, hidden_size, vocab_size, sequence_length=25, learning_rate=0.01):
        """
        Initialize the Character RNN model.

        Args:
            hidden_size: Size of the hidden layer
            vocab_size: Size of the vocabulary (number of unique characters)
            sequence_length: Length of input sequences for training
            learning_rate: Learning rate for weight updates
        """
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Initialize model parameters
        # Xavier/Glorot initialization
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

        # Memory variables for Adagrad optimization
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

        # Loss tracking
        self.smooth_loss = -np.log(1.0 / vocab_size) * sequence_length

    def forward(self, inputs, hprev):
        """
        Forward pass through the network.

        Args:
            inputs: List of input character indices
            hprev: Initial hidden state

        Returns:
            xs: Input one-hot vectors
            hs: Hidden states
            ys: Output vectors (unnormalized)
            ps: Softmax probabilities
            h_last: Final hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)

        # Forward pass
        for t in range(len(inputs)):
            # One-hot encode the input character
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1

            # Hidden state
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh
            )

            # Output
            ys[t] = np.dot(self.Why, hs[t]) + self.by

            # Softmax probabilities
            exp_y = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_y / np.sum(exp_y)

        return xs, hs, ys, ps, hs[len(inputs) - 1]

    def backward(self, xs, hs, ps, targets):
        """
        Backward pass to compute gradients.

        Args:
            xs: Input one-hot vectors
            hs: Hidden states
            ps: Softmax probabilities
            targets: Target character indices

        Returns:
            dWxh, dWhh, dWhy: Gradients for weights
            dbh, dby: Gradients for biases
            h_last: Final hidden state
        """
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Backpropagation through time
        dhnext = np.zeros_like(hs[0])

        # Calculate loss
        loss = 0
        for t in reversed(range(len(targets))):
            # Cross-entropy loss
            loss += -np.log(ps[t][targets[t], 0])

            # Gradient of softmax output
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            # Gradient of Why
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # Gradient of hidden state
            dh = np.dot(self.Why.T, dy) + dhnext

            # Gradient of tanh
            dhraw = (1 - hs[t] * hs[t]) * dh

            # Gradient of biases
            dbh += dhraw

            # Gradient of Whh
            dWhh += np.dot(dhraw, hs[t - 1].T)

            # Gradient of Wxh
            dWxh += np.dot(dhraw, xs[t].T)

            # Gradient for next iteration
            dhnext = np.dot(self.Whh.T, dhraw)

        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(targets) - 1]

    def optimize(self, dWxh, dWhh, dWhy, dbh, dby):
        """
        Update model parameters using Adagrad optimization.

        Args:
            dWxh, dWhh, dWhy: Gradients for weights
            dbh, dby: Gradients for biases
        """
        # Adagrad update rule
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby],
        ):
            mem += dparam * dparam
            param -= self.learning_rate * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_idx, n, char_to_idx, idx_to_char, temperature=1.0):
        """
        Sample a sequence of characters from the trained model.

        Args:
            h: Hidden state
            seed_idx: Index of the seed character
            n: Number of characters to generate
            char_to_idx: Mapping from characters to indices
            idx_to_char: Mapping from indices to characters
            temperature: Sampling temperature (higher = more random)

        Returns:
            generated_text: Generated character sequence
        """
        # Initialize hidden state and first character
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1

        # List to store generated characters
        indices = [seed_idx]

        # Generate characters
        for i in range(n):
            # Forward pass for a single step
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by

            # Apply temperature to control randomness
            y = y / temperature

            # Convert to probabilities
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            # Sample from the distribution
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())

            # Prepare input for next step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            indices.append(idx)

        # Convert indices to characters
        generated_text = "".join([idx_to_char[idx] for idx in indices])

        return generated_text

    def train(
        self,
        data,
        char_to_idx,
        idx_to_char,
        num_iterations=100000,
        print_every=1000,
        sample_every=1000,
    ):
        """
        Train the character-level language model.

        Args:
            data: Training text
            char_to_idx: Mapping from characters to indices
            idx_to_char: Mapping from indices to characters
            num_iterations: Number of training iterations
            print_every: How often to print progress
            sample_every: How often to sample from the model

        Returns:
            loss_history: List of loss values during training
        """
        loss_history = []
        iterations = []

        # Convert training data to indices
        data_size = len(data)
        data_indices = [char_to_idx[ch] for ch in data]

        # Initial hidden state
        hprev = np.zeros((self.hidden_size, 1))

        # Training loop
        for i in range(num_iterations):
            # Prepare training data for this iteration
            # Start at a random position in the data
            p = random.randint(0, data_size - self.sequence_length - 1)
            inputs = data_indices[p : p + self.sequence_length]
            targets = data_indices[p + 1 : p + self.sequence_length + 1]

            # Forward pass
            xs, hs, ys, ps, hprev = self.forward(inputs, hprev)

            # Backward pass
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.backward(xs, hs, ps, targets)

            # Update parameters
            self.optimize(dWxh, dWhh, dWhy, dbh, dby)

            # Update smooth loss for monitoring
            self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * loss

            # Print progress
            if i % print_every == 0:
                print(f"Iteration {i}, Loss: {self.smooth_loss:.4f}")
                loss_history.append(self.smooth_loss)
                iterations.append(i)

            # Sample from the model periodically
            if i % sample_every == 0:
                sample_idx = random.randint(0, self.vocab_size - 1)
                sample_text = self.sample(
                    hprev, sample_idx, 200, char_to_idx, idx_to_char
                )
                print(f"\nSample at iteration {i}:\n{sample_text}\n")

        return iterations, loss_history


def fetch_text_data(source="shakespeare"):
    """
    Fetch text data for training the language model.

    Args:
        source: Source of the text data ('shakespeare', 'wiki', or 'custom')

    Returns:
        text: Text data as a string
    """
    data_dir = os.path.join("exercises", "module3", "data")
    os.makedirs(data_dir, exist_ok=True)

    if source == "shakespeare":
        # Shakespeare's works
        file_path = os.path.join(data_dir, "shakespeare.txt")

        if not os.path.exists(file_path):
            print("Downloading Shakespeare's text...")
            # Shakespeare's complete works from Project Gutenberg
            url = "https://www.gutenberg.org/files/100/100-0.txt"
            response = requests.get(url)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract a portion to make training faster for demonstration
        start_idx = text.find("THE SONNETS")
        end_idx = text.find("THE END", start_idx)
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]

    elif source == "custom":
        # Use a small custom text for testing
        text = """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """

    else:
        raise ValueError(f"Unknown source: {source}")

    return text


def preprocess_text(text):
    """
    Preprocess text for training.

    Args:
        text: Input text

    Returns:
        processed_text: Processed text
        char_to_idx: Mapping from characters to indices
        idx_to_char: Mapping from indices to characters
    """
    # Remove some special characters if needed
    processed_text = text

    # Create vocabulary (unique characters in the text)
    chars = sorted(list(set(processed_text)))
    vocab_size = len(chars)

    print(f"Vocabulary size: {vocab_size} unique characters")

    # Create mappings from characters to indices and vice versa
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    return processed_text, char_to_idx, idx_to_char, vocab_size


def visualize_loss(iterations, loss_history):
    """
    Visualize training loss.

    Args:
        iterations: List of iteration numbers
        loss_history: List of loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_temperatures(
    model, seed_text, char_to_idx, idx_to_char, temps=[0.5, 0.7, 1.0, 1.2]
):
    """
    Compare text generation with different sampling temperatures.

    Args:
        model: Trained CharRNN model
        seed_text: Seed text to start generation
        char_to_idx: Mapping from characters to indices
        idx_to_char: Mapping from indices to characters
        temps: List of temperature values to compare
    """
    # Initialize with the last character of the seed text
    seed_char = seed_text[-1]
    h = np.zeros((model.hidden_size, 1))

    # Generate text with different temperatures
    print("\nComparing different sampling temperatures:")

    for temp in temps:
        generated = model.sample(
            h, char_to_idx[seed_char], 200, char_to_idx, idx_to_char, temperature=temp
        )
        print(f"\nTemperature: {temp}")
        print("=" * 40)
        print(seed_text + generated[1:])  # Skip the first character which is the seed
        print("=" * 40)


def main():
    """Main function to run the character-level language model exercise."""
    print("===== Character-Level Language Model Exercise =====")

    # 1. Fetch and preprocess text data
    print("\n1. Fetching text data...")
    text = fetch_text_data(source="custom")  # Use 'shakespeare' for full dataset

    print("\n2. Preprocessing text...")
    processed_text, char_to_idx, idx_to_char, vocab_size = preprocess_text(text)

    # Show a sample of the text
    print(f"\nSample of the text:\n{processed_text[:500]}...")

    # 2. Create and train the model
    print("\n3. Creating and training the model...")

    # Hyperparameters
    hidden_size = 100
    sequence_length = 25
    learning_rate = 0.01
    num_iterations = 10000  # Reduced for demonstration, use 100000+ for better results

    # Create model
    model = CharRNN(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
    )

    # Train model
    iterations, loss_history = model.train(
        processed_text,
        char_to_idx,
        idx_to_char,
        num_iterations=num_iterations,
        print_every=1000,
        sample_every=2000,
    )

    # 3. Visualize training results
    print("\n4. Visualizing training results...")
    visualize_loss(iterations, loss_history)

    # 4. Generate text with different temperatures
    print("\n5. Generating text with different temperatures...")
    seed_text = "To be, or not to be"
    compare_temperatures(
        model, seed_text, char_to_idx, idx_to_char, temps=[0.5, 0.7, 1.0, 1.5]
    )

    print("\nThis exercise demonstrated:")
    print("- How to build a character-level language model using RNNs")
    print("- The process of training on text data")
    print("- How sampling temperature affects text generation")
    print("- The creative capabilities of neural networks")


if __name__ == "__main__":
    main()
