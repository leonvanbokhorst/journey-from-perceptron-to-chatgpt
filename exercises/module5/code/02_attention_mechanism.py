#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise: Sequence-to-Sequence Model with Attention

This exercise implements a Seq2Seq model with an attention mechanism.
The attention mechanism allows the decoder to focus on different parts
of the input sequence at each decoding step.

The exercise demonstrates:
1. Implementation of Bahdanau attention mechanism
2. Integration of attention in the Seq2Seq architecture
3. Visualization of attention weights
4. Comparison with basic Seq2Seq model (without attention)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import random
import string
import time

# Import basic Seq2Seq model from previous exercise
try:
    from exercises.module5.code.01_seq2seq_basics import Encoder as BasicEncoder
    from exercises.module5.code.01_seq2seq_basics import one_hot_encode, generate_simple_translation_dataset
except ImportError:
    print("Error importing from previous exercise. Make sure 01_seq2seq_basics.py is in the same directory.")
    # Simplified versions of these functions would be defined here in a real implementation


class EncoderWithAttention(BasicEncoder):
    """
    Encoder part of the Seq2Seq model with attention.
    
    Similar to the basic encoder but returns all hidden states for attention.
    """
    
    def forward(self, x_sequence: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run the encoder forward, keeping all hidden states for attention.
        
        Args:
            x_sequence: Input sequence of shape (seq_len, input_size, 1)
            
        Returns:
            h: Final hidden state
            hs: List of all hidden states for attention
        """
        # Same implementation as basic encoder, but we explicitly return all hidden states
        h = np.zeros((self.hidden_size, 1))
        hs = []
        
        self.last_inputs = x_sequence
        self.last_hs = [h]
        
        for x in x_sequence:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hs.append(h)
            self.last_hs.append(h)
        
        return h, hs


class AttentionMechanism:
    """
    Bahdanau attention mechanism.
    
    Uses the current decoder hidden state and all encoder hidden states
    to compute attention weights.
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize the attention mechanism.
        
        Args:
            hidden_size: Size of the hidden states
        """
        self.hidden_size = hidden_size
        
        # Parameter initialization for attention
        # Score function: v^T * tanh(W1 * h_decoder + W2 * h_encoder)
        self.W1 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v = np.random.randn(hidden_size, 1) * 0.01
        
        # Store for backpropagation
        self.last_attn_weights = None
        self.last_encoder_hs = None
        self.last_decoder_h = None
    
    def compute_attention_weights(self, decoder_h: np.ndarray, 
                                 encoder_hs: List[np.ndarray]) -> np.ndarray:
        """
        Compute attention weights using the attention score function.
        
        Args:
            decoder_h: Current decoder hidden state
            encoder_hs: All encoder hidden states
            
        Returns:
            attention_weights: Normalized attention weights
        """
        # Store for backpropagation
        self.last_decoder_h = decoder_h
        self.last_encoder_hs = encoder_hs
        
        # Calculate attention scores for each encoder hidden state
        scores = []
        for h_enc in encoder_hs:
            # Attention score: v^T * tanh(W1 * h_decoder + W2 * h_encoder)
            score = np.dot(self.v.T, np.tanh(np.dot(self.W1, decoder_h) + np.dot(self.W2, h_enc)))
            scores.append(float(score))
        
        # Convert to numpy array
        scores = np.array(scores)
        
        # Apply softmax to get attention weights
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Store for backpropagation and visualization
        self.last_attn_weights = attention_weights
        
        return attention_weights
    
    def compute_context_vector(self, encoder_hs: List[np.ndarray], 
                               attention_weights: np.ndarray) -> np.ndarray:
        """
        Compute context vector as weighted sum of encoder hidden states.
        
        Args:
            encoder_hs: All encoder hidden states
            attention_weights: Attention weights for each encoder hidden state
            
        Returns:
            context_vector: Weighted sum of encoder hidden states
        """
        # Initialize context vector
        context_vector = np.zeros_like(encoder_hs[0])
        
        # Compute weighted sum
        for i, h_enc in enumerate(encoder_hs):
            context_vector += h_enc * attention_weights[i]
        
        return context_vector


class DecoderWithAttention:
    """
    Decoder with attention mechanism.
    
    Uses attention to focus on different parts of the input sequence at each time step.
    """
    
    def __init__(self, hidden_size: int, output_size: int):
        """
        Initialize the decoder with attention.
        
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
        # Context to hidden
        self.Wch = np.random.randn(hidden_size, hidden_size) * 0.01
        # Hidden to output
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        # Initialize attention mechanism
        self.attention = AttentionMechanism(hidden_size)
        
        # For storing forward pass values
        self.last_hs = None
        self.last_inputs = None
        self.last_outputs = None
        self.last_contexts = None
        self.last_attention_weights = []
    
    def forward(self, h0: np.ndarray, encoder_hs: List[np.ndarray],
                target_sequence: Optional[np.ndarray] = None,
                max_len: int = 100, teacher_forcing: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Run the decoder with attention forward.
        
        Args:
            h0: Initial hidden state (from encoder)
            encoder_hs: All encoder hidden states for attention
            target_sequence: Target sequence for teacher forcing
            max_len: Maximum generation length
            teacher_forcing: Whether to use teacher forcing
            
        Returns:
            ys: Output vectors
            hs: Hidden states
            attention_weights: Attention weights for each time step
        """
        # Reset state
        self.last_hs = [h0]
        self.last_inputs = []
        self.last_outputs = []
        self.last_contexts = []
        self.last_attention_weights = []
        
        # Initialize state
        h = h0
        y = np.zeros((self.output_size, 1))
        
        # Initialize output lists
        ys = []
        hs = []
        attention_weights = []
        
        # Determine sequence length
        if teacher_forcing and target_sequence is not None:
            seq_len = len(target_sequence)
        else:
            seq_len = max_len
        
        for t in range(seq_len):
            # Store input
            self.last_inputs.append(y)
            
            # Compute attention weights
            attn_weights = self.attention.compute_attention_weights(h, encoder_hs)
            attention_weights.append(attn_weights)
            self.last_attention_weights.append(attn_weights)
            
            # Compute context vector
            context = self.attention.compute_context_vector(encoder_hs, attn_weights)
            self.last_contexts.append(context)
            
            # Update hidden state with attention context
            h = np.tanh(np.dot(self.Whh, h) + np.dot(self.Wyh, y) + np.dot(self.Wch, context) + self.bh)
            hs.append(h)
            self.last_hs.append(h)
            
            # Compute output
            y = np.dot(self.Why, h) + self.by
            
            # Apply softmax
            y_exp = np.exp(y - np.max(y))
            y = y_exp / np.sum(y_exp)
            
            ys.append(y)
            self.last_outputs.append(y)
            
            # If teacher forcing, use target as next input
            if teacher_forcing and target_sequence is not None and t < len(target_sequence) - 1:
                y = target_sequence[t+1]
        
        return ys, hs, attention_weights
    
    def backward(self, dys: List[np.ndarray], learning_rate: float = 0.01) -> np.ndarray:
        """
        Backpropagate through the decoder with attention.
        
        Args:
            dys: Gradients of loss with respect to outputs
            learning_rate: Learning rate for parameter updates
            
        Returns:
            dh0: Gradient of loss with respect to initial hidden state
        """
        # Initialize gradients
        dWyh = np.zeros_like(self.Wyh)
        dWhh = np.zeros_like(self.Whh)
        dWch = np.zeros_like(self.Wch)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Initialize context and hidden state gradients
        dh_next = np.zeros_like(self.last_hs[0])
        
        # Backpropagate through time
        for t in reversed(range(len(dys))):
            # Gradient from output
            dy = dys[t]
            dWhy += np.dot(dy, self.last_hs[t+1].T)
            dby += dy
            
            # Gradient to hidden state
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # Gradient through tanh
            dtanh = (1 - self.last_hs[t+1] * self.last_hs[t+1]) * dh
            
            # Gradients for weights
            dbh += dtanh
            dWhh += np.dot(dtanh, self.last_hs[t].T)
            dWyh += np.dot(dtanh, self.last_inputs[t].T)
            dWch += np.dot(dtanh, self.last_contexts[t].T)
            
            # Gradient for next timestep
            dh_next = np.dot(self.Whh.T, dtanh)
            
            # Gradients for attention mechanism (simplified)
            # In a full implementation, we would backpropagate through the attention mechanism
        
        # Clip gradients
        for grad in [dWyh, dWhh, dWch, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        # Update parameters
        self.Wyh -= learning_rate * dWyh
        self.Whh -= learning_rate * dWhh
        self.Wch -= learning_rate * dWch
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return dh_next


class Seq2SeqWithAttention:
    """
    Sequence-to-Sequence model with attention mechanism.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Seq2Seq model with attention.
        
        Args:
            input_size: Size of input vectors
            hidden_size: Size of hidden states
            output_size: Size of output vectors
        """
        self.encoder = EncoderWithAttention(input_size, hidden_size)
        self.decoder = DecoderWithAttention(hidden_size, output_size)
    
    def forward(self, x_sequence: np.ndarray, target_sequence: Optional[np.ndarray] = None,
                teacher_forcing: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run the Seq2Seq model with attention forward.
        
        Args:
            x_sequence: Input sequence
            target_sequence: Target sequence for teacher forcing
            teacher_forcing: Whether to use teacher forcing
            
        Returns:
            outputs: Generated sequence
            attention_weights: Attention weights for visualization
        """
        # Encode input sequence
        _, encoder_hs = self.encoder.forward(x_sequence)
        
        # Decode with attention
        outputs, _, attention_weights = self.decoder.forward(
            encoder_hs[-1], encoder_hs, target_sequence, teacher_forcing=teacher_forcing)
        
        return outputs, attention_weights
    
    def backward(self, d_outputs: List[np.ndarray], learning_rate: float = 0.01) -> None:
        """
        Backpropagate through the Seq2Seq model with attention.
        
        Args:
            d_outputs: Gradients of loss with respect to outputs
            learning_rate: Learning rate for parameter updates
        """
        # Backpropagate through decoder
        dh = self.decoder.backward(d_outputs, learning_rate)
        
        # Backpropagate through encoder
        self.encoder.backward(dh, learning_rate)
    
    def train(self, X: List[np.ndarray], y: List[np.ndarray], epochs: int = 100,
              learning_rate: float = 0.01, print_every: int = 10) -> List[float]:
        """
        Train the Seq2Seq model with attention.
        
        Args:
            X: Input sequences
            y: Target sequences
            epochs: Number of training epochs
            learning_rate: Learning rate
            print_every: How often to print progress
            
        Returns:
            losses: Training losses
        """
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(len(X)):
                # Forward pass with teacher forcing
                outputs, _ = self.forward(X[i], y[i], teacher_forcing=True)
                
                # Compute loss and gradients
                loss = 0
                dy_sequence = []
                
                for t in range(len(outputs)):
                    # Cross entropy loss
                    y_pred = outputs[t]
                    y_true = y[i][t] if t < len(y[i]) else np.zeros_like(y_pred)
                    
                    # Compute loss
                    loss -= np.sum(y_true * np.log(y_pred + 1e-10))
                    
                    # Compute gradient
                    dy = y_pred - y_true
                    dy_sequence.append(dy)
                
                # Normalize loss
                loss /= len(outputs)
                epoch_loss += loss
                
                # Backward pass
                self.backward(dy_sequence, learning_rate)
            
            # Average loss
            epoch_loss /= len(X)
            losses.append(epoch_loss)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        
        return losses


def translate_with_attention(model: Seq2SeqWithAttention, input_sequence: np.ndarray,
                            input_vocab: List[str], target_vocab: List[str],
                            max_length: int = 20) -> Tuple[str, np.ndarray]:
    """
    Translate an input sequence using the Seq2Seq model with attention.
    
    Args:
        model: Trained model
        input_sequence: Input sequence
        input_vocab: Input vocabulary
        target_vocab: Target vocabulary
        max_length: Maximum output length
        
    Returns:
        translation: Translated string
        attention_matrix: Attention weights for visualization
    """
    # Forward pass without teacher forcing
    outputs, attention_weights = model.forward(input_sequence, teacher_forcing=False)
    
    # Convert outputs to indices
    indices = [np.argmax(output) for output in outputs]
    
    # Convert indices to characters
    translation = ''.join([target_vocab[idx] for idx in indices])
    
    # Stop at end token
    if '.' in translation:
        translation = translation[:translation.index('.')]
    
    # Convert attention weights to matrix
    attention_matrix = np.vstack(attention_weights)
    
    return translation, attention_matrix


def visualize_translation_with_attention(input_str: str, output_str: str, attention_matrix: np.ndarray) -> None:
    """
    Visualize attention weights for a translation.
    
    Args:
        input_str: Input string
        output_str: Output string (translation)
        attention_matrix: Attention weights matrix
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot attention matrix
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)
    
    # Set labels
    ax.set_xticklabels([''] + list(input_str), rotation=90)
    ax.set_yticklabels([''] + list(output_str))
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    plt.title('Attention Visualization')
    plt.show()


def compare_models_on_translation() -> None:
    """
    Compare Seq2Seq models with and without attention on a translation task.
    """
    # Generate dataset
    input_sequences, target_sequences, input_vocab, target_vocab = generate_simple_translation_dataset(
        num_examples=1000, max_length=10)
    
    # Convert to one-hot encoding
    X = [one_hot_encode(seq, len(input_vocab)) for seq in input_sequences]
    y = [one_hot_encode(seq, len(target_vocab)) for seq in target_sequences]
    
    # Split into train/test
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Model parameters
    input_size = len(input_vocab)
    hidden_size = 128
    output_size = len(target_vocab)
    
    # Initialize models
    print("Initializing models...")
    from exercises.module5.code.01_seq2seq_basics import Seq2Seq
    seq2seq = Seq2Seq(input_size, hidden_size, output_size)
    seq2seq_attn = Seq2SeqWithAttention(input_size, hidden_size, output_size)
    
    # Train models (with fewer epochs for demonstration)
    print("\nTraining basic Seq2Seq model...")
    basic_losses = seq2seq.train(X_train[:200], y_train[:200], epochs=30, learning_rate=0.01, print_every=5)
    
    print("\nTraining Seq2Seq model with attention...")
    attn_losses = seq2seq_attn.train(X_train[:200], y_train[:200], epochs=30, learning_rate=0.01, print_every=5)
    
    # Compare training losses
    plt.figure(figsize=(10, 5))
    plt.plot(basic_losses, label='Basic Seq2Seq')
    plt.plot(attn_losses, label='Seq2Seq with Attention')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Test and visualize a few examples
    print("\nTesting and visualizing translations with attention:")
    for i in range(3):
        # Get test example
        idx = i
        
        # Get input and target text
        input_seq = input_sequences[800 + idx]
        input_text = ''.join([input_vocab[idx] for idx in input_seq])
        
        target_seq = target_sequences[800 + idx]
        target_text = ''.join([target_vocab[idx] for idx in target_seq])
        
        # Translate with basic model
        basic_translation = translate(seq2seq, X_test[idx], input_vocab, target_vocab)
        
        # Translate with attention model
        attn_translation, attention_matrix = translate_with_attention(
            seq2seq_attn, X_test[idx], input_vocab, target_vocab)
        
        # Print results
        print(f"\nExample {i+1}:")
        print(f"Input:                 {input_text}")
        print(f"Target:                {target_text}")
        print(f"Basic Seq2Seq:         {basic_translation}")
        print(f"Seq2Seq with Attention: {attn_translation}")
        
        # Visualize attention
        visualize_translation_with_attention(input_text, attn_translation, attention_matrix)


def main() -> None:
    """Main function to demonstrate Seq2Seq with attention."""
    print("Starting Sequence-to-Sequence with Attention demonstration...")
    
    # Compare basic Seq2Seq with attention-based Seq2Seq
    compare_models_on_translation()
    
    print("\nExercise completed! You've implemented a Seq2Seq model with attention.")
    
    print("\nExercises for the reader:")
    print("1. Experiment with different attention mechanisms (e.g., dot product attention).")
    print("2. Implement a bidirectional encoder to improve context representation.")
    print("3. Try a more complex translation task with real language data.")
    print("4. Compare performance when increasing sequence lengths (attention helps with longer sequences).")


if __name__ == "__main__":
    main() 