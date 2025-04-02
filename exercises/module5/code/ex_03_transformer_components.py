#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise: Transformer Components Implementation

This exercise implements the key components of the Transformer architecture,
focusing on multi-head attention and positional encoding. These components
are crucial building blocks for the full Transformer model.

The exercise demonstrates:
1. Implementation of scaled dot-product attention
2. Implementation of multi-head attention
3. Implementation of positional encoding
4. A simplified encoder block
5. Visualization of attention patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
import math

# Set random seed for reproducibility
np.random.seed(42)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        query: Query matrix of shape (batch_size, seq_len_q, depth)
        key: Key matrix of shape (batch_size, seq_len_k, depth)
        value: Value matrix of shape (batch_size, seq_len_k, depth_v)
        mask: Optional mask matrix of shape (batch_size, seq_len_q, seq_len_k)
              to mask out certain positions

    Returns:
        output: Attention output of shape (batch_size, seq_len_q, depth_v)
        attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
    """
    # Compute attention scores: matmul(Q, K.T) / sqrt(d_k)
    depth = query.shape[-1]
    matmul_qk = np.matmul(query, np.transpose(key, (0, 2, 1)))

    # Scale by sqrt(depth)
    scaled_attention_scores = matmul_qk / np.sqrt(depth)

    # Apply mask if provided (add large negative value to masked positions)
    if mask is not None:
        scaled_attention_scores += mask * -1e9

    # Apply softmax for attention weights
    attention_weights = np.exp(
        scaled_attention_scores
        - np.max(scaled_attention_scores, axis=-1, keepdims=True)
    )
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True) + 1e-9

    # Compute attention output: matmul(attention_weights, V)
    output = np.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention implementation.

    Splits input into multiple heads to attend to different parts of the sequence.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
        """
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        # Initialize weight matrices
        self.wq = np.random.randn(d_model, d_model) * 0.01
        self.wk = np.random.randn(d_model, d_model) * 0.01
        self.wv = np.random.randn(d_model, d_model) * 0.01
        self.wo = np.random.randn(d_model, d_model) * 0.01

        # For storing intermediate values
        self.last_input_q = None
        self.last_input_k = None
        self.last_input_v = None
        self.last_attention_weights = None

    def split_heads(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Split the last dimension into (num_heads, depth).
        Transpose to (batch_size, num_heads, seq_len, depth)

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Batch size

        Returns:
            Split tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # Reshape to (batch_size, seq_len, num_heads, depth)
        x_split = x.reshape(batch_size, -1, self.num_heads, self.depth)

        # Transpose to (batch_size, num_heads, seq_len, depth)
        return np.transpose(x_split, (0, 2, 1, 3))

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.

        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor

        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights for visualization
        """
        batch_size = q.shape[0]

        # Store inputs for backward pass
        self.last_input_q = q
        self.last_input_k = k
        self.last_input_v = v

        # Linear projections and split heads
        q_projected = np.matmul(q, self.wq)
        k_projected = np.matmul(k, self.wk)
        v_projected = np.matmul(v, self.wv)

        q_split = self.split_heads(q_projected, batch_size)
        k_split = self.split_heads(k_projected, batch_size)
        v_split = self.split_heads(v_projected, batch_size)

        # Apply scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask
        )

        # Store attention weights for visualization
        self.last_attention_weights = attention_weights

        # Reshape back: (batch_size, seq_len, d_model)
        # First transpose: (batch_size, seq_len, num_heads, depth)
        scaled_attention_transposed = np.transpose(scaled_attention, (0, 2, 1, 3))

        # Then reshape: (batch_size, seq_len, d_model)
        concat_attention = scaled_attention_transposed.reshape(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = np.matmul(concat_attention, self.wo)

        return output, attention_weights

    def backward(
        self, d_output: np.ndarray, learning_rate: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of multi-head attention.

        Args:
            d_output: Gradient from the next layer
            learning_rate: Learning rate for parameter updates

        Returns:
            dq, dk, dv: Gradients for q, k, v inputs
        """
        # This is a simplified backward pass for demonstration purposes
        # A complete implementation would compute proper gradients for all parameters

        batch_size = d_output.shape[0]

        # Gradient for output projection
        d_concat_attention = np.matmul(d_output, self.wo.T)
        dwo = np.matmul(np.transpose(self.last_input_v, (0, 2, 1)), d_output)

        # Update output weights
        self.wo -= learning_rate * np.mean(dwo, axis=0)

        # Simplified: just propagate gradients back to inputs
        dq = np.matmul(d_concat_attention, self.wq.T)
        dk = np.matmul(d_concat_attention, self.wk.T)
        dv = np.matmul(d_concat_attention, self.wv.T)

        # Update input weights
        dwq = np.matmul(np.transpose(self.last_input_q, (0, 2, 1)), d_concat_attention)
        dwk = np.matmul(np.transpose(self.last_input_k, (0, 2, 1)), d_concat_attention)
        dwv = np.matmul(np.transpose(self.last_input_v, (0, 2, 1)), d_concat_attention)

        self.wq -= learning_rate * np.mean(dwq, axis=0)
        self.wk -= learning_rate * np.mean(dwk, axis=0)
        self.wv -= learning_rate * np.mean(dwv, axis=0)

        return dq, dk, dv


def get_positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Compute positional encoding as per the original Transformer paper.

    Args:
        seq_length: Length of the sequence
        d_model: Dimension of the model (embedding size)

    Returns:
        pos_encoding: Positional encoding matrix of shape (1, seq_length, d_model)
    """
    # Initialize positional encoding matrix
    pos_encoding = np.zeros((1, seq_length, d_model))

    # Compute position indices and dimension indices
    positions = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Compute sine and cosine values
    pos_encoding[0, :, 0::2] = np.sin(positions * div_term)
    pos_encoding[0, :, 1::2] = np.cos(positions * div_term)

    return pos_encoding


class FeedForwardNetwork:
    """
    Feed-forward network used in Transformer.

    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize the feed-forward network.

        Args:
            d_model: Model dimension (input and output size)
            d_ff: Hidden dimension of the feed-forward network
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights
        self.w1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.w2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))

        # For storing intermediate values
        self.last_input = None
        self.last_hidden = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of same shape as input
        """
        # Store input for backward pass
        self.last_input = x

        # First linear transformation
        hidden = np.matmul(x, self.w1) + self.b1

        # ReLU activation
        hidden_relu = np.maximum(0, hidden)

        # Store hidden state for backward pass
        self.last_hidden = hidden_relu

        # Second linear transformation
        output = np.matmul(hidden_relu, self.w2) + self.b2

        return output

    def backward(self, d_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass of feed-forward network.

        Args:
            d_output: Gradient from the next layer
            learning_rate: Learning rate for parameter updates

        Returns:
            dx: Gradient with respect to input
        """
        # Gradient for second linear transformation
        d_hidden_relu = np.matmul(d_output, self.w2.T)
        dw2 = np.matmul(np.transpose(self.last_hidden, (0, 2, 1)), d_output)
        db2 = np.sum(d_output, axis=(0, 1), keepdims=True)

        # Gradient through ReLU
        d_hidden = d_hidden_relu * (self.last_hidden > 0)

        # Gradient for first linear transformation
        dx = np.matmul(d_hidden, self.w1.T)
        dw1 = np.matmul(np.transpose(self.last_input, (0, 2, 1)), d_hidden)
        db1 = np.sum(d_hidden, axis=(0, 1), keepdims=True)

        # Update parameters
        self.w1 -= learning_rate * np.mean(dw1, axis=0)
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * np.mean(dw2, axis=0)
        self.b2 -= learning_rate * db2

        return dx


class LayerNormalization:
    """
    Layer normalization for Transformer.

    Normalizes the last dimension of the input tensor.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model: Dimension to normalize
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))

        # For storing intermediate values
        self.last_input = None
        self.last_mean = None
        self.last_var = None
        self.last_normalized = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Normalized output of same shape as input
        """
        # Store input for backward pass
        self.last_input = x

        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Store for backward pass
        self.last_mean = mean
        self.last_var = var

        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        self.last_normalized = x_normalized

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        return output

    def backward(self, d_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass of layer normalization.

        Args:
            d_output: Gradient from the next layer
            learning_rate: Learning rate for parameter updates

        Returns:
            dx: Gradient with respect to input
        """
        # Gradient for scale and shift
        dgamma = np.sum(d_output * self.last_normalized, axis=(0, 1), keepdims=True)
        dbeta = np.sum(d_output, axis=(0, 1), keepdims=True)

        # Update parameters
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

        # Gradient for normalized input
        dx_normalized = d_output * self.gamma

        # Simplified: propagate gradient through normalization
        # This is a simplified implementation; full backprop through layer norm is more complex
        batch_size, seq_len, _ = self.last_input.shape
        dx = dx_normalized / np.sqrt(self.last_var + self.eps)

        return dx


class EncoderLayer:
    """
    Transformer encoder layer.

    Consists of multi-head attention and feed-forward network with layer normalization.
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1
    ):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout_rate: Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # Layer normalization
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask for padding

        Returns:
            output: Output tensor of same shape as input
            attention_weights: Attention weights for visualization
        """
        # Multi-head attention with residual connection and layer normalization
        attn_output, attention_weights = self.mha.forward(x, x, x, mask)

        # Apply dropout (simplified: just scale by dropout keep rate during training)
        if self.dropout_rate > 0:
            attn_output *= 1 - self.dropout_rate

        # First residual connection and layer normalization
        norm1_output = self.layernorm1.forward(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn.forward(norm1_output)

        # Apply dropout
        if self.dropout_rate > 0:
            ffn_output *= 1 - self.dropout_rate

        # Second residual connection and layer normalization
        output = self.layernorm2.forward(norm1_output + ffn_output)

        return output, attention_weights

    def backward(self, d_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Backward pass of encoder layer.

        Args:
            d_output: Gradient from the next layer
            learning_rate: Learning rate for parameter updates

        Returns:
            dx: Gradient with respect to input
        """
        # Backward through second layer norm and residual
        d_layernorm2 = self.layernorm2.backward(d_output, learning_rate)
        d_ffn_output = d_layernorm2
        d_norm1_output_res = d_layernorm2

        # Backward through feed-forward network
        d_norm1_output = self.ffn.backward(d_ffn_output, learning_rate)

        # Add gradient from residual connection
        d_norm1_output += d_norm1_output_res

        # Backward through first layer norm and residual
        d_layernorm1 = self.layernorm1.backward(d_norm1_output, learning_rate)
        d_attn_output = d_layernorm1
        d_x_res = d_layernorm1

        # Backward through multi-head attention
        d_q, d_k, d_v = self.mha.backward(d_attn_output, learning_rate)

        # Add gradient from residual connection
        dx = d_x_res + d_q

        return dx


def visualize_positional_encoding(max_length: int = 100, d_model: int = 512) -> None:
    """
    Visualize the positional encoding.

    Args:
        max_length: Maximum sequence length
        d_model: Model dimension
    """
    # Get positional encoding
    pos_encoding = get_positional_encoding(max_length, d_model)

    # Visualize a subset of the positional encoding
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(pos_encoding[0, :, :100], cmap="RdBu")
    plt.xlabel("Depth")
    plt.ylabel("Position")
    plt.colorbar()
    plt.title("Positional Encoding Visualization (first 100 dimensions)")
    plt.show()

    # Plot a few positions across all dimensions
    plt.figure(figsize=(12, 6))
    for pos in [1, 10, 20, 50, 99]:
        plt.plot(pos_encoding[0, pos, :], label=f"Position {pos}")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Value")
    plt.title("Positional Encoding Values at Different Positions")
    plt.grid(True)
    plt.show()


def visualize_attention_patterns() -> None:
    """Visualize different attention patterns."""
    seq_len = 10
    d_model = 64
    batch_size = 1

    # Create some sample inputs
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)

    # Create different attention masks
    # 1. No mask (full attention)
    no_mask = None

    # 2. Causal mask (for decoder self-attention)
    causal_mask = np.triu(np.ones((batch_size, seq_len, seq_len)), k=1) * -1e9

    # 3. Random padding mask
    padding_mask = np.ones((batch_size, seq_len, seq_len))
    padding_mask[:, :, -3:] = -1e9  # Mask out the last 3 positions

    # Compute attention with different masks
    _, full_attn = scaled_dot_product_attention(q, k, v, no_mask)
    _, causal_attn = scaled_dot_product_attention(q, k, v, causal_mask)
    _, padding_attn = scaled_dot_product_attention(q, k, v, padding_mask)

    # Visualize attention patterns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Full attention
    axes[0].matshow(full_attn[0], cmap="viridis")
    axes[0].set_title("Full Attention")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")

    # Causal attention
    axes[1].matshow(causal_attn[0], cmap="viridis")
    axes[1].set_title("Causal Attention (Decoder Self-Attention)")
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")

    # Padding mask attention
    axes[2].matshow(padding_attn[0], cmap="viridis")
    axes[2].set_title("Attention with Padding Mask")
    axes[2].set_xlabel("Key Position")
    axes[2].set_ylabel("Query Position")

    plt.tight_layout()
    plt.show()


def test_multi_head_attention() -> None:
    """Test and demonstrate multi-head attention."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    # Create test inputs
    x = np.random.randn(batch_size, seq_len, d_model)

    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)

    # Forward pass
    print("Running multi-head attention forward pass...")
    start_time = time.time()
    output, attention_weights = mha.forward(x, x, x)
    end_time = time.time()

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Forward pass time: {end_time - start_time:.4f} seconds")

    # Visualize attention weights for the first head and first batch
    plt.figure(figsize=(8, 6))
    plt.matshow(attention_weights[0, 0], cmap="viridis")
    plt.title("Attention Weights (First Head, First Batch)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    plt.show()


def test_encoder_layer() -> None:
    """Test and demonstrate encoder layer."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    d_ff = 256

    # Create test input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Initialize encoder layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # Forward pass
    print("Running encoder layer forward pass...")
    start_time = time.time()
    output, attention_weights = encoder_layer.forward(x)
    end_time = time.time()

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Forward pass time: {end_time - start_time:.4f} seconds")

    # Backward pass (testing gradient flow)
    print("\nRunning encoder layer backward pass...")
    start_time = time.time()
    d_output = np.random.randn(*output.shape)
    dx = encoder_layer.backward(d_output, learning_rate=0.01)
    end_time = time.time()

    print(f"Gradient shape: {dx.shape}")
    print(f"Backward pass time: {end_time - start_time:.4f} seconds")


def main() -> None:
    """Main function to demonstrate Transformer components."""
    print("Starting Transformer Components demonstration...")

    # Visualize positional encoding
    print("\nVisualizing positional encoding...")
    visualize_positional_encoding(max_length=100, d_model=128)

    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    visualize_attention_patterns()

    # Test multi-head attention
    print("\nTesting multi-head attention...")
    test_multi_head_attention()

    # Test encoder layer
    print("\nTesting encoder layer...")
    test_encoder_layer()

    print(
        "\nExercise completed! You've implemented key components of the Transformer architecture."
    )

    print("\nExercises for the reader:")
    print("1. Experiment with different positional encoding schemes.")
    print("2. Implement a complete Transformer encoder and decoder.")
    print("3. Train a simplified Transformer model on a sequence task.")
    print("4. Explore different attention mechanisms and compare their performance.")


if __name__ == "__main__":
    main()
