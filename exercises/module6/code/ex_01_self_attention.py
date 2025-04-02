#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 1: Implementing Scaled Dot-Product Attention

This exercise covers:
1. Implementing the scaled dot-product attention mechanism
2. Visualizing attention patterns between sequence elements
3. Experimenting with different attention masks

The self-attention mechanism is the core building block of the Transformer architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    Compute scaled dot-product attention.

    Args:
        q: Queries tensor of shape (batch_size, num_heads, seq_len_q, d_k)
        k: Keys tensor of shape (batch_size, num_heads, seq_len_k, d_k)
        v: Values tensor of shape (batch_size, num_heads, seq_len_v, d_v) where seq_len_v = seq_len_k
        mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)
        dropout: Optional dropout layer

    Returns:
        Attention output and attention weights
    """
    # Calculate attention scores
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Apply attention weights to values
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention layer that can be used as a building block for the Transformer.
    """

    def __init__(self, d_model, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len)

        Returns:
            Self-attention output and attention weights
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        q = self.query(x)  # (batch_size, seq_len, d_model)
        k = self.key(x)  # (batch_size, seq_len, d_model)
        v = self.value(x)  # (batch_size, seq_len, d_model)

        # Reshape for scaled dot-product attention
        q = q.view(batch_size, 1, seq_len, self.d_model)  # Add head dimension
        k = k.view(batch_size, 1, seq_len, self.d_model)
        v = v.view(batch_size, 1, seq_len, self.d_model)

        # Compute self-attention
        output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout
        )

        # Reshape output
        output = output.view(batch_size, seq_len, self.d_model)

        return output, attention_weights


def create_padding_mask(seq, pad_idx=0):
    """
    Create a mask to hide padding tokens.

    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_idx: Index used for padding

    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    # Mask is 1 for tokens that are not padding and 0 for padding tokens
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(seq_len):
    """
    Create a mask to prevent attention to future tokens (for decoder).

    Args:
        seq_len: Length of the sequence

    Returns:
        Look-ahead mask tensor of shape (1, 1, seq_len, seq_len)
    """
    # Create a lower triangular matrix with 1s
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
    return mask


def visualize_attention(attention_weights, tokens=None, title="Attention Weights"):
    """
    Visualize attention weights using a heatmap.

    Args:
        attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        tokens: Optional list of tokens for axis labels
        title: Title for the plot
    """
    # Take the first batch and first head, convert to numpy
    weights = attention_weights[0, 0].detach().numpy()

    # Create figure
    plt.figure(figsize=(8, 6))

    # Create heatmap
    ax = sns.heatmap(
        weights, annot=True, cmap="YlGnBu", xticklabels=tokens, yticklabels=tokens
    )

    # Set labels and title
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.title(title)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Example usage
def self_attention_example():
    """
    Example demonstrating self-attention on a small sequence.
    """
    print("Self-Attention Example")
    print("-" * 50)

    # Create a simple sequence of token embeddings
    tokens = ["I", "love", "transformers", "they", "are", "powerful"]
    seq_len = len(tokens)
    d_model = 8  # Small embedding size for demonstration

    # Create random embeddings for the tokens
    embeddings = torch.randn(1, seq_len, d_model)  # (batch_size=1, seq_len, d_model)

    # Create self-attention layer
    self_attention = SelfAttention(d_model)

    # Apply self-attention
    output, attention_weights = self_attention(embeddings)

    # Print shapes
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize attention weights
    print("\nVisualization of attention patterns:")
    visualize_attention(attention_weights, tokens, "Self-Attention Weights")


def masked_attention_example():
    """
    Example demonstrating masked self-attention.
    """
    print("\nMasked Self-Attention Example")
    print("-" * 50)

    # Create a sequence of token embeddings
    tokens = ["I", "love", "transformers", "they", "are", "powerful"]
    seq_len = len(tokens)
    d_model = 8  # Small embedding size for demonstration

    # Create random embeddings for the tokens
    embeddings = torch.randn(1, seq_len, d_model)

    # Create self-attention layer
    self_attention = SelfAttention(d_model)

    # Create a look-ahead mask (causal mask)
    mask = create_look_ahead_mask(seq_len)

    # Apply masked self-attention
    output, attention_weights = self_attention(embeddings, mask)

    # Visualize attention weights
    print("Visualization of masked attention patterns:")
    print("(Each token can only attend to itself and previous tokens)")
    visualize_attention(attention_weights, tokens, "Masked Self-Attention Weights")


def padding_mask_example():
    """
    Example demonstrating padding mask.
    """
    print("\nPadding Mask Example")
    print("-" * 50)

    # Create a padded sequence
    # Assume 0 is the padding index
    tokens = ["I", "love", "transformers", "<pad>", "<pad>", "<pad>"]
    seq = torch.tensor([[1, 2, 3, 0, 0, 0]])  # Batch size 1, seq_len 6, with padding
    seq_len = seq.size(1)
    d_model = 8

    # Create random embeddings for the tokens
    embeddings = torch.randn(1, seq_len, d_model)

    # Create self-attention layer
    self_attention = SelfAttention(d_model)

    # Create padding mask
    mask = create_padding_mask(seq)

    # Apply masked self-attention
    output, attention_weights = self_attention(embeddings, mask)

    # Visualize attention weights
    print("Visualization of attention with padding mask:")
    print("(Tokens cannot attend to padding tokens)")
    visualize_attention(attention_weights, tokens, "Self-Attention with Padding Mask")


def exercise_custom_attention_pattern():
    """
    Exercise: Implement a custom attention pattern.

    In this exercise, you'll create a custom attention pattern where tokens
    can only attend to positions at a specific distance from themselves.
    """
    print("\nExercise: Custom Attention Pattern")
    print("-" * 50)
    print("Implement a custom attention pattern where tokens can only attend")
    print("to positions at a specific distance (e.g., ±1) from themselves.")

    tokens = ["The", "transformer", "model", "revolutionized", "NLP", "research"]
    seq_len = len(tokens)
    d_model = 8

    # Create random embeddings
    embeddings = torch.randn(1, seq_len, d_model)

    # Create a custom mask where each token can only attend to
    # itself and its immediate neighbors (±1 position)
    # TODO: Implement this mask
    # Hint: Start with a matrix of zeros and fill in ones at allowed positions
    custom_mask = torch.zeros(1, 1, seq_len, seq_len)

    # For each position, allow attention to itself and immediate neighbors
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= 1:  # Self and immediate neighbors
                custom_mask[0, 0, i, j] = 1

    # Create self-attention layer
    self_attention = SelfAttention(d_model)

    # Apply with custom mask
    output, attention_weights = self_attention(embeddings, custom_mask)

    # Visualize
    visualize_attention(
        attention_weights, tokens, "Custom Attention Pattern (±1 Position)"
    )

    print("\nBonus: Try modifying the mask to create different attention patterns!")


if __name__ == "__main__":
    # Check if seaborn is available, if not use matplotlib only
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not found, using matplotlib for visualization")

        # Function to use with matplotlib only
        def visualize_attention(
            attention_weights, tokens=None, title="Attention Weights"
        ):
            weights = attention_weights[0, 0].detach().numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(weights, cmap="YlGnBu")
            plt.colorbar()
            if tokens:
                plt.xticks(range(len(tokens)), tokens, rotation=45)
                plt.yticks(range(len(tokens)), tokens)
            plt.xlabel("Key tokens")
            plt.ylabel("Query tokens")
            plt.title(title)
            plt.tight_layout()
            plt.show()

    # Run the examples
    self_attention_example()
    masked_attention_example()
    padding_mask_example()
    exercise_custom_attention_pattern()

    print("\nCongratulations! You've implemented and explored self-attention, the core")
    print(
        "building block of the Transformer architecture. In the next exercise, you'll build"
    )
    print("on this foundation to implement multi-head attention.")
