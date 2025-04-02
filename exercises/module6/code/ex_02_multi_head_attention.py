#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 2: Implementing Multi-Head Attention

This exercise covers:
1. Building a multi-head attention module
2. Understanding how multiple attention heads capture different relationships
3. Analyzing and visualizing what different heads learn

Multi-head attention allows the model to jointly attend to information from different
representation subspaces at different positions, providing a richer model of relationships.
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


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module splits the queries, keys, and values into multiple heads,
    applies scaled dot-product attention to each head independently,
    and then combines the results.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear projections for Q, K, V
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Final output projection
        self.wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model) where seq_len_v = seq_len_k
            mask: Optional mask tensor

        Returns:
            Output tensor and attention weights
        """
        batch_size = q.size(0)

        # Linear projections and reshape for multi-head attention
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention to each head
        attn_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout
        )

        # Reshape and apply final projection
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.wo(attn_output)

        return output, attention_weights


class SelfMultiHeadAttention(nn.Module):
    """
    Self Multi-Head Attention module.

    This is a wrapper around the MultiHeadAttention module that uses the same input
    for queries, keys, and values (self-attention).
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SelfMultiHeadAttention, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for self multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor and attention weights
        """
        return self.mha(x, x, x, mask)


def create_look_ahead_mask(seq_len):
    """
    Create a mask to prevent attention to future tokens (for decoder).

    Args:
        seq_len: Length of the sequence

    Returns:
        Look-ahead mask tensor of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
    return mask


def visualize_attention_heads(
    attention_weights, tokens=None, title="Multi-Head Attention Weights"
):
    """
    Visualize attention weights for multiple heads.

    Args:
        attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        tokens: Optional list of tokens for axis labels
        title: Title for the plot
    """
    batch_size, num_heads, seq_len_q, seq_len_k = attention_weights.shape

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4))

    if num_heads == 1:
        axes = [axes]  # Make it iterable if there's only one head

    # Plot each head's attention pattern
    for h, ax in enumerate(axes):
        # Get attention weights for this head
        weights = attention_weights[0, h].detach().numpy()

        # Create heatmap
        im = ax.imshow(weights, cmap="YlGnBu")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set labels
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)

        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.set_title(f"Head {h+1}")

    # Set overall title
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the suptitle
    plt.show()


def simple_example():
    """
    Example demonstrating multi-head attention on a simple sequence.
    """
    print("Multi-Head Attention Example")
    print("-" * 50)

    # Create a simple sequence
    tokens = ["I", "love", "transformers", "they", "are", "powerful"]
    seq_len = len(tokens)
    d_model = 16  # Small embedding size for demonstration
    num_heads = 4  # 4 attention heads

    # Create random embeddings for the tokens
    embeddings = torch.randn(1, seq_len, d_model)  # (batch_size=1, seq_len, d_model)

    # Create multi-head self-attention layer
    mha = SelfMultiHeadAttention(d_model, num_heads)

    # Apply multi-head self-attention
    output, attention_weights = mha(embeddings)

    # Print shapes
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize attention weights for all heads
    print("\nVisualization of attention patterns for each head:")
    visualize_attention_heads(attention_weights, tokens, "Multi-Head Self-Attention")

    print("\nNotice how different heads can focus on different patterns!")


def compare_single_vs_multi_head():
    """
    Compare single-head attention with multi-head attention.
    """
    print("\nComparing Single-Head vs. Multi-Head Attention")
    print("-" * 50)

    # Parameters
    tokens = ["The", "transformer", "architecture", "uses", "self", "attention"]
    seq_len = len(tokens)
    d_model = 16

    # Create random embeddings
    embeddings = torch.randn(1, seq_len, d_model)

    # Create single-head and multi-head attention layers
    single_head = SelfMultiHeadAttention(d_model, num_heads=1)
    multi_head = SelfMultiHeadAttention(d_model, num_heads=4)

    # Apply attention
    single_output, single_attn_weights = single_head(embeddings)
    multi_output, multi_attn_weights = multi_head(embeddings)

    # Compare performance
    # Let's define a simple task: predict the next token (we'll just use MSE with the original)
    single_loss = F.mse_loss(single_output[:, :-1, :], embeddings[:, 1:, :])
    multi_loss = F.mse_loss(multi_output[:, :-1, :], embeddings[:, 1:, :])

    print("Task: Predict the next token embedding")
    print(f"Single-head attention loss: {single_loss.item():.6f}")
    print(f"Multi-head attention loss: {multi_loss.item():.6f}")

    # Visualize attention patterns
    print("\nSingle-Head Attention Pattern:")
    visualize_attention_heads(single_attn_weights, tokens, "Single-Head Attention")

    print("\nMulti-Head Attention Patterns:")
    visualize_attention_heads(multi_attn_weights, tokens, "Multi-Head Attention")

    print(
        "\nNote: Multi-head attention allows different heads to focus on different relationships"
    )
    print("in the data, providing a richer representation.")


def masked_multi_head_example():
    """
    Example demonstrating masked multi-head attention (as used in decoders).
    """
    print("\nMasked Multi-Head Attention Example")
    print("-" * 50)

    # Create a sequence
    tokens = ["The", "Transformer", "model", "was", "introduced", "in"]
    seq_len = len(tokens)
    d_model = 16
    num_heads = 4

    # Create random embeddings
    embeddings = torch.randn(1, seq_len, d_model)

    # Create multi-head self-attention layer
    mha = SelfMultiHeadAttention(d_model, num_heads)

    # Create a look-ahead mask (causal mask)
    mask = create_look_ahead_mask(seq_len)

    # Apply masked multi-head self-attention
    output, attention_weights = mha(embeddings, mask)

    # Visualize attention weights
    print("Visualization of masked multi-head attention patterns:")
    print("(Each token can only attend to itself and previous tokens)")
    visualize_attention_heads(
        attention_weights, tokens, "Masked Multi-Head Self-Attention"
    )

    print("\nThis type of masking is used in the Transformer decoder to ensure")
    print(
        "that predictions for position i can only depend on known outputs at positions less than i."
    )


def exercise_head_specialization():
    """
    Exercise: Analyze head specialization.

    In this exercise, you'll analyze how different attention heads might specialize
    in capturing different types of relationships in the data.
    """
    print("\nExercise: Head Specialization")
    print("-" * 50)

    # Create a sequence with clear positional patterns
    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    seq_len = len(tokens)
    d_model = 32
    num_heads = 4

    # Instead of random embeddings, we'll create embeddings with some structure
    # For simplicity, we'll use a simple positional pattern: even and odd positions
    embeddings = torch.zeros(1, seq_len, d_model)

    # For even positions, set the first half of dimensions to 1
    # For odd positions, set the second half of dimensions to 1
    for i in range(seq_len):
        if i % 2 == 0:  # Even position
            embeddings[0, i, : d_model // 2] = 1.0
        else:  # Odd position
            embeddings[0, i, d_model // 2 :] = 1.0

    # Add some noise for variability
    embeddings += torch.randn(1, seq_len, d_model) * 0.1

    # Create multi-head self-attention layer
    mha = SelfMultiHeadAttention(d_model, num_heads)

    # Apply multi-head self-attention
    output, attention_weights = mha(embeddings)

    # Visualize attention patterns
    visualize_attention_heads(
        attention_weights, tokens, "Head Specialization in Multi-Head Attention"
    )

    print("\nAnalysis:")
    print("Look for patterns in the attention weights of each head.")
    print(
        "Some heads might focus on local relationships (attending to adjacent tokens),"
    )
    print(
        "while others might focus on global patterns or specific syntactic relationships."
    )
    print("\nTry to identify what each head is learning:")
    print("- Is any head focusing on adjacent tokens (local attention)?")
    print("- Is any head attending uniformly across all tokens (global attention)?")
    print("- Is any head showing specialized attention to particular positions?")

    # Calculate the average attention distance for each head
    distances = []
    for h in range(num_heads):
        weights = attention_weights[0, h].detach()

        # Create distance matrix
        pos_i, pos_j = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len))
        dist_matrix = torch.abs(pos_i - pos_j).float()

        # Calculate weighted average distance
        avg_dist = torch.sum(weights * dist_matrix) / torch.sum(weights)
        distances.append(avg_dist.item())

    print("\nAverage attention distance by head:")
    for h, dist in enumerate(distances):
        print(f"Head {h+1}: {dist:.2f}")

    print("\nHeads with smaller average distance focus more on local context,")
    print("while heads with larger average distance attend more globally.")


def exercise_cross_attention():
    """
    Exercise: Implement cross-attention.

    In this exercise, you'll implement and experiment with cross-attention,
    where queries come from one sequence and keys/values from another.
    This is used in the Transformer decoder to attend to the encoder outputs.
    """
    print("\nExercise: Cross-Attention")
    print("-" * 50)

    # Create source and target sequences
    src_tokens = ["I", "love", "machine", "learning"]
    tgt_tokens = ["J'", "adore", "l'", "apprentissage", "automatique"]

    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)
    d_model = 16
    num_heads = 2

    # Create random embeddings for source and target
    src_embeddings = torch.randn(1, src_len, d_model)
    tgt_embeddings = torch.randn(1, tgt_len, d_model)

    # Create multi-head attention layer for cross-attention
    cross_attention = MultiHeadAttention(d_model, num_heads)

    # Apply cross-attention (queries from target, keys/values from source)
    # This simulates the encoder-decoder attention in the Transformer
    output, attention_weights = cross_attention(
        tgt_embeddings, src_embeddings, src_embeddings
    )

    # Visualize cross-attention weights
    print("Cross-attention from target (French) to source (English):")

    # Custom visualization for cross-attention (different sequence lengths)
    batch_size, num_heads, tgt_seq_len, src_seq_len = attention_weights.shape

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 5, 5))

    if num_heads == 1:
        axes = [axes]

    # Plot each head's attention pattern
    for h, ax in enumerate(axes):
        weights = attention_weights[0, h].detach().numpy()

        im = ax.imshow(weights, cmap="YlGnBu")
        plt.colorbar(im, ax=ax)

        # Set labels
        ax.set_xticks(range(src_len))
        ax.set_yticks(range(tgt_len))
        ax.set_xticklabels(src_tokens, rotation=45)
        ax.set_yticklabels(tgt_tokens)

        ax.set_xlabel("Source (English)")
        ax.set_ylabel("Target (French)")
        ax.set_title(f"Head {h+1}")

    plt.suptitle("Cross-Attention Weights")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    print("\nAnalysis:")
    print("In cross-attention, each target token attends to source tokens.")
    print(
        "This allows the model to use information from the encoder when generating the output."
    )
    print(
        "For example, in machine translation, this is how the model aligns target words"
    )
    print("with their corresponding source words.")

    print("\nObservation Task:")
    print("Look at the attention patterns and try to identify if there's any alignment")
    print("between words that have similar meanings in the two languages.")
    print("For example, does 'adore' attend strongly to 'love'?")


if __name__ == "__main__":
    # Check if seaborn is available, if not use matplotlib only
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not found, using matplotlib for visualization")
        # This won't replace our existing visualization functions,
        # but the plots will be less pretty without seaborn

    # Run the examples
    simple_example()
    compare_single_vs_multi_head()
    masked_multi_head_example()

    # Run the exercises
    exercise_head_specialization()
    exercise_cross_attention()

    print("\nCongratulations! You've implemented and explored multi-head attention,")
    print("a key component of the Transformer architecture. In the next exercise,")
    print("you'll build a complete Transformer encoder block using these components.")
