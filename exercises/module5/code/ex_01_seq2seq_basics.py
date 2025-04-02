#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise: Basic Sequence-to-Sequence Model Implementation

This exercise implements a basic sequence-to-sequence (Seq2Seq) model
without attention mechanisms. The model consists of an encoder and a decoder,
both using RNN/LSTM cells.

The exercise demonstrates:
1. Basic encoder-decoder architecture
2. Implementation of a Seq2Seq model from scratch using NumPy
3. Teacher forcing during training
4. Inference using greedy decoding
5. Application to a simple translation task
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Any
import random
import string
import time

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class Encoder:
    """
    Encoder for Sequence-to-Sequence model.

    Takes an input sequence and encodes it into a context vector.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize the Encoder.

        Args:
            input_size: Dimension of input vectors
            hidden_size: Dimension of hidden state vectors
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        # Input to hidden
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        # Hidden to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Biases
        self.bh = np.zeros((hidden_size, 1))

        # For storing the values from the forward pass
        self.last_inputs = None
        self.last_hs = None

    def forward(self, x_sequence: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run the encoder forward on the input sequence.

        Args:
            x_sequence: Input sequence of shape (seq_len, input_size, 1)

        Returns:
            h: Final hidden state
            hs: List of all hidden states
        """
        # Initialize the hidden state
        h = np.zeros((self.hidden_size, 1))
        hs = []

        # Store inputs and hidden states for backpropagation
        self.last_inputs = x_sequence
        self.last_hs = [h]

        # Process the sequence
        for x in x_sequence:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hs.append(h)
            self.last_hs.append(h)

        # Return the final hidden state and all hidden states
        return h, hs

    def backward(self, dh_next: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Backpropagate through the encoder.

        Args:
            dh_next: Gradient of the loss with respect to the next hidden state
            learning_rate: Learning rate for gradient descent

        Returns:
            dx: Gradient of the loss with respect to the input
        """
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dh = dh_next

        # Backpropagate through time
        for t in reversed(range(len(self.last_inputs))):
            # Get stored values
            h_prev = self.last_hs[t]
            h = self.last_hs[t + 1]
            x = self.last_inputs[t]

            # Gradient through tanh
            dtanh = (1 - h * h) * dh

            # Update gradients
            dbh += dtanh
            dWxh += np.dot(dtanh, x.T)
            dWhh += np.dot(dtanh, h_prev.T)

            # Gradient for next timestep
            dh = np.dot(self.Whh.T, dtanh)

        # Clip gradients to mitigate exploding gradients
        for grad in [dWxh, dWhh, dbh]:
            np.clip(grad, -5, 5, out=grad)

        # Update parameters
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.bh -= learning_rate * dbh

        return dh


class Decoder:
    """
    Decoder for Sequence-to-Sequence model.

    Takes a context vector from the encoder and generates output sequence.
    """

    def __init__(self, hidden_size: int, output_size: int):
        """
        Initialize the Decoder.

        Args:
            hidden_size: Dimension of hidden state vectors
            output_size: Dimension of output vectors
        """
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        # Hidden to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Input to hidden (previous output to hidden)
        self.Wyh = np.random.randn(hidden_size, output_size) * 0.01
        # Hidden to output
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # For storing the forward pass values
        self.last_hs = None
        self.last_inputs = None
        self.last_outputs = None

    def forward(
        self,
        h0: np.ndarray,
        target_sequence: Optional[np.ndarray] = None,
        max_len: int = 100,
        teacher_forcing: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run the decoder forward.

        Args:
            h0: Initial hidden state (context vector from encoder)
            target_sequence: Target sequence for teacher forcing (if None, use generated output)
            max_len: Maximum length of the generated sequence (if not using teacher forcing)
            teacher_forcing: Whether to use teacher forcing during training

        Returns:
            ys: List of output vectors
            hs: List of hidden states
        """
        # Reset state
        self.last_hs = [h0]
        self.last_inputs = []
        self.last_outputs = []

        # Initialize hidden state with the context vector
        h = h0

        # Initialize output with zeros
        y = np.zeros((self.output_size, 1))

        # Store outputs and hidden states
        ys = []
        hs = []

        # Determine sequence length
        if teacher_forcing and target_sequence is not None:
            seq_len = len(target_sequence)
        else:
            seq_len = max_len

        # Generate the sequence
        for t in range(seq_len):
            # Store previous output
            self.last_inputs.append(y)

            # Update hidden state
            h = np.tanh(np.dot(self.Whh, h) + np.dot(self.Wyh, y) + self.bh)
            hs.append(h)
            self.last_hs.append(h)

            # Compute output
            y = np.dot(self.Why, h) + self.by

            # Apply softmax for probability distribution
            y_exp = np.exp(y - np.max(y))  # For numerical stability
            y = y_exp / np.sum(y_exp)

            ys.append(y)
            self.last_outputs.append(y)

            # If teacher forcing, use target as next input
            if (
                teacher_forcing
                and target_sequence is not None
                and t < len(target_sequence) - 1
            ):
                y = target_sequence[t + 1]

        return ys, hs

    def backward(
        self, dys: List[np.ndarray], learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        Backpropagate through the decoder.

        Args:
            dys: List of gradients of loss with respect to decoder outputs
            learning_rate: Learning rate for gradient descent

        Returns:
            dh0: Gradient of loss with respect to initial hidden state
        """
        # Initialize gradients
        dWhh = np.zeros_like(self.Whh)
        dWyh = np.zeros_like(self.Wyh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Initialize hidden state gradient
        dh_next = np.zeros_like(self.last_hs[0])

        # Backpropagate through time
        for t in reversed(range(len(dys))):
            # Gradient from output
            dy = dys[t]
            dWhy += np.dot(dy, self.last_hs[t + 1].T)
            dby += dy

            # Gradient to hidden state
            dh = np.dot(self.Why.T, dy) + dh_next

            # Gradient through tanh
            dtanh = (1 - self.last_hs[t + 1] * self.last_hs[t + 1]) * dh

            # Update gradients
            dbh += dtanh
            dWhh += np.dot(dtanh, self.last_hs[t].T)
            dWyh += np.dot(dtanh, self.last_inputs[t].T)

            # Gradient for next timestep
            dh_next = np.dot(self.Whh.T, dtanh)

        # Clip gradients to mitigate exploding gradients
        for grad in [dWyh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        # Update parameters
        self.Whh -= learning_rate * dWhh
        self.Wyh -= learning_rate * dWyh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        return dh_next


class Seq2Seq:
    """
    Sequence-to-Sequence model composed of an encoder and a decoder.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Seq2Seq model.

        Args:
            input_size: Dimension of input vectors to the encoder
            hidden_size: Dimension of hidden state vectors
            output_size: Dimension of output vectors from the decoder
        """
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(
        self,
        x_sequence: np.ndarray,
        target_sequence: Optional[np.ndarray] = None,
        teacher_forcing: bool = True,
    ) -> List[np.ndarray]:
        """
        Run the Seq2Seq model forward.

        Args:
            x_sequence: Input sequence
            target_sequence: Target sequence (for teacher forcing)
            teacher_forcing: Whether to use teacher forcing during training

        Returns:
            y_sequence: Generated output sequence
        """
        # Encode input sequence
        context_vector, _ = self.encoder.forward(x_sequence)

        # Decode from context vector
        y_sequence, _ = self.decoder.forward(
            context_vector, target_sequence, teacher_forcing=teacher_forcing
        )

        return y_sequence

    def backward(
        self, d_outputs: List[np.ndarray], learning_rate: float = 0.01
    ) -> None:
        """
        Backpropagate through the Seq2Seq model.

        Args:
            d_outputs: Gradients of loss with respect to outputs
            learning_rate: Learning rate for gradient descent
        """
        # Backpropagate through decoder
        dh = self.decoder.backward(d_outputs, learning_rate)

        # Backpropagate through encoder
        self.encoder.backward(dh, learning_rate)

    def train(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        epochs: int = 100,
        learning_rate: float = 0.01,
        print_every: int = 10,
    ) -> List[float]:
        """
        Train the Seq2Seq model.

        Args:
            X: List of input sequences
            y: List of target sequences
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            print_every: How often to print progress

        Returns:
            losses: List of training losses
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            for i in range(len(X)):
                # Forward pass with teacher forcing
                outputs = self.forward(X[i], y[i], teacher_forcing=True)

                # Compute loss
                loss = 0
                dy_sequence = []

                for t in range(len(outputs)):
                    # Cross entropy loss
                    y_pred = outputs[t]
                    y_true = y[i][t] if t < len(y[i]) else np.zeros_like(y_pred)

                    # Compute loss
                    loss -= np.sum(y_true * np.log(y_pred + 1e-10))

                    # Gradient of loss with respect to output
                    dy = y_pred - y_true
                    dy_sequence.append(dy)

                # Normalize the loss
                loss /= len(outputs)
                epoch_loss += loss

                # Backward pass
                self.backward(dy_sequence, learning_rate)

            # Average loss across all examples
            epoch_loss /= len(X)
            losses.append(epoch_loss)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        return losses


def one_hot_encode(sequence: List[int], vocab_size: int) -> np.ndarray:
    """
    One-hot encode a sequence of integers.

    Args:
        sequence: List of integer indices
        vocab_size: Size of the vocabulary

    Returns:
        encoded: One-hot encoded sequence of shape (len(sequence), vocab_size, 1)
    """
    encoded = np.zeros((len(sequence), vocab_size, 1))
    for i, idx in enumerate(sequence):
        encoded[i, idx, 0] = 1
    return encoded


def generate_simple_translation_dataset(
    num_examples: int = 1000, max_length: int = 10
) -> Tuple[List[List[int]], List[List[int]], List[str], List[str]]:
    """
    Generate a simple translation dataset from "gibberish" to reversed "gibberish".

    Args:
        num_examples: Number of examples to generate
        max_length: Maximum sequence length

    Returns:
        input_sequences: List of input sequences (as lists of indices)
        target_sequences: List of target sequences (as lists of indices)
        input_vocab: Input vocabulary
        target_vocab: Target vocabulary
    """
    # Create vocabulary
    input_vocab = list(string.ascii_lowercase + " .")
    target_vocab = input_vocab.copy()

    # Generate samples
    input_sequences = []
    target_sequences = []

    for _ in range(num_examples):
        # Generate random "gibberish" words
        length = random.randint(3, max_length)
        input_text = "".join(
            random.choice(string.ascii_lowercase) for _ in range(length)
        )

        # "Translation" = reverse the input + special token
        target_text = input_text[::-1]

        # Convert to sequences of indices
        input_seq = [input_vocab.index(c) for c in input_text] + [
            input_vocab.index(".")
        ]
        # Add start and end tokens for target (using "." as both for simplicity)
        target_seq = (
            [target_vocab.index(".")]
            + [target_vocab.index(c) for c in target_text]
            + [target_vocab.index(".")]
        )

        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    return input_sequences, target_sequences, input_vocab, target_vocab


def visualize_attention(
    input_text: str, translations: List[str], attention_matrix: np.ndarray
) -> None:
    """
    Visualize attention weights.

    Args:
        input_text: Input text
        translations: Translations (could be just one)
        attention_matrix: Attention weights matrix of shape (output_len, input_len)
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the attention weights
    cax = ax.matshow(attention_matrix, cmap="viridis")

    # Set labels
    ax.set_xticklabels([""] + list(input_text), rotation=90)
    ax.set_yticklabels([""] + list(translations[0]))

    # Add colorbar
    fig.colorbar(cax)

    # Set title
    ax.set_title("Attention weights for translation: " + translations[0])

    # Show the plot
    plt.tight_layout()
    plt.show()


def translate(
    model: Seq2Seq,
    input_sequence: np.ndarray,
    input_vocab: List[str],
    target_vocab: List[str],
    max_length: int = 20,
) -> str:
    """
    Translate an input sequence using the trained model.

    Args:
        model: Trained Seq2Seq model
        input_sequence: Input sequence (one-hot encoded)
        input_vocab: Input vocabulary
        target_vocab: Target vocabulary
        max_length: Maximum length of the generated sequence

    Returns:
        translation: Translated string
    """
    # Forward pass without teacher forcing (inference mode)
    outputs = model.forward(input_sequence, teacher_forcing=False)

    # Convert outputs to indices
    indices = [np.argmax(output) for output in outputs]

    # Convert indices to characters
    translation = "".join([target_vocab[idx] for idx in indices])

    # Stop at the end token (".")
    if "." in translation:
        translation = translation[: translation.index(".")]

    return translation


def main() -> None:
    """Main function to demonstrate the Seq2Seq model."""
    print("Generating simple translation dataset...")
    input_sequences, target_sequences, input_vocab, target_vocab = (
        generate_simple_translation_dataset(num_examples=1000, max_length=10)
    )

    # Convert sequences to one-hot encoding
    X = [one_hot_encode(seq, len(input_vocab)) for seq in input_sequences]
    y = [one_hot_encode(seq, len(target_vocab)) for seq in target_sequences]

    # Initialize the model
    input_size = len(input_vocab)
    hidden_size = 128
    output_size = len(target_vocab)
    seq2seq = Seq2Seq(input_size, hidden_size, output_size)

    # Train the model
    print("Training the Seq2Seq model...")
    start_time = time.time()
    losses = seq2seq.train(
        X[:800], y[:800], epochs=50, learning_rate=0.01, print_every=5
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Test the model
    print("\nTesting translation examples:")
    for i in range(5):
        idx = 800 + i  # Use examples from the test set

        # Original input sequence
        input_seq = input_sequences[idx]
        input_text = "".join([input_vocab[idx] for idx in input_seq])

        # True target sequence
        target_seq = target_sequences[idx]
        target_text = "".join([target_vocab[idx] for idx in target_seq])

        # Model translation
        translation = translate(seq2seq, X[idx], input_vocab, target_vocab)

        # Print results
        print(f"Input:      {input_text}")
        print(f"True:       {target_text}")
        print(f"Translated: {translation}")
        print()

    print(
        "Exercise completed! You've implemented a basic Seq2Seq model without attention."
    )

    # Exercises for the reader
    print("\nExercises for the reader:")
    print(
        "1. Try different hyperparameters to see how they affect the model's performance."
    )
    print("2. Modify the model to use GRU or LSTM cells instead of simple RNN cells.")
    print("3. Implement beam search for decoding during inference.")
    print("4. Create a more complex translation task and see how the model performs.")


if __name__ == "__main__":
    main()
