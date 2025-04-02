#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 3: Implementing the Transformer Encoder

This exercise covers:
1. Implementing a Transformer encoder block
2. Building a stack of encoder layers
3. Adding positional encoding
4. Processing sequences and visualizing representations

The Transformer encoder is a powerful module for contextualizing sequences
by using self-attention and feed-forward networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module from Exercise 2.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections and reshape
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attn_output = torch.matmul(attention_weights, v)

        # Reshape and apply final projection
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.wo(attn_output)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    This consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    This normalizes the inputs across the features, applying an affine transformation.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """
    Positional Encoding.

    This adds positional information to the input embeddings using sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    Each encoder layer consists of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network

    Each sub-layer has a residual connection and is followed by layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask for padding tokens

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-Attention sub-layer
        attn_output, self_attn_weights = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)  # Residual connection and layer norm

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)  # Residual connection and layer norm

        return x, self_attn_weights


class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder.

    This consists of an embedding layer, positional encoding, and a stack of encoder layers.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.1,
        max_len=5000,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: Integer tensor of shape (batch_size, seq_len) containing token indices
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Convert token indices to embeddings
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply dropout
        x = self.dropout(x)

        # Process through the stack of encoder layers
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, mask)
            attentions.append(attention)

        # Apply final layer normalization
        x = self.norm(x)

        return x, attentions


def visualize_positional_encoding(d_model=32, max_len=100):
    """
    Visualize the positional encodings.
    """
    pos_enc = PositionalEncoding(d_model, max_len)

    # Create a tensor of zeros
    x = torch.zeros(1, max_len, d_model)

    # Apply positional encoding (this will just add the positional encoding to zeros)
    pe = pos_enc(x).squeeze(0).detach().numpy()

    # Create a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(pe, cmap="viridis", aspect="auto")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Position")
    plt.title("Positional Encoding")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Plot a few dimensions
    plt.figure(figsize=(10, 6))
    for i in range(0, d_model, 8):  # Plot every 8th dimension
        plt.plot(pe[:, i], label=f"Dim {i}")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.title("Positional Encoding Dimensions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Observations:")
    print("1. Each row represents the encoding for a position in the sequence.")
    print("2. Each column represents a dimension of the encoding.")
    print("3. The encoding pattern has a specific frequency for each dimension.")
    print(
        "4. These frequencies vary, allowing the model to distinguish different positions."
    )
    print(
        "5. Positions closer together have similar encodings, while distant positions differ more."
    )


def create_padding_mask(seq, pad_idx=0):
    """
    Create a mask to hide padding tokens.

    Args:
        seq: Tensor of shape (batch_size, seq_len)
        pad_idx: Padding token index

    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def transformer_encoder_example():
    """
    Example demonstrating the usage of a Transformer encoder.
    """
    print("Transformer Encoder Example")
    print("-" * 50)

    # Parameters
    vocab_size = 1000
    d_model = 32
    num_heads = 4
    d_ff = 128
    num_layers = 2

    # Create a sample input sequence
    batch_size = 2
    seq_len = 8

    # Input sequences (token indices)
    x = torch.randint(
        1, vocab_size, (batch_size, seq_len)
    )  # Random token indices (avoid pad token 0)

    # Create a padding mask
    mask = create_padding_mask(x)

    # Create a Transformer encoder
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)

    # Apply encoder
    encoded, attentions = encoder(x, mask)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Encoded output shape: {encoded.shape}")
    print(f"Number of attention layers: {len(attentions)}")
    print(f"Attention weights shape (per layer): {attentions[0].shape}")


def analyze_representations(
    vocab_size=1000, d_model=32, num_heads=4, d_ff=128, num_layers=2
):
    """
    Analyze how the encoder transforms token representations through the layers.
    """
    print("\nAnalyzing Token Representations Through Encoder Layers")
    print("-" * 50)

    # Create a sequence with meaningful tokens
    # Let's imagine these token ids correspond to:
    tokens = [
        "The",
        "transformer",
        "model",
        "revolutionized",
        "natural",
        "language",
        "processing",
    ]
    token_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)  # (1, 7)

    # Create a simple encoder
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)

    # Record embeddings at different stages
    # We'll modify the forward method to capture intermediate representations

    # Initial embeddings (after positional encoding)
    emb = encoder.embedding(token_ids) * math.sqrt(d_model)
    emb_with_pos = encoder.positional_encoding(emb)

    # List to store representations after each layer
    layer_outputs = [emb_with_pos.detach()]

    # Process through encoder layers
    x = emb_with_pos
    for layer in encoder.layers:
        x, _ = layer(x)
        layer_outputs.append(x.detach())

    # Final output after normalization
    final_output = encoder.norm(x).detach()
    layer_outputs.append(final_output)

    # Analyze the representations
    # Compute cosine similarity between token representations at each layer
    print("Analyzing how token representations evolve:")

    # For simplicity, let's focus on the first token
    token_idx = 0

    # Convert representations to unit vectors for cosine similarity
    unit_vectors = []
    for output in layer_outputs:
        vec = output[0, token_idx].numpy()
        unit_vectors.append(vec / np.linalg.norm(vec))

    # Compute cosine similarity between consecutive layers
    for i in range(1, len(unit_vectors)):
        sim = np.dot(unit_vectors[i - 1], unit_vectors[i])
        print(f"Cosine similarity between layer {i-1} and {i}: {sim:.4f}")

    # Visualize one of the attention layers
    # For simplicity, let's use the first attention layer
    _, attn_weights = encoder.layers[0].self_attn(
        emb_with_pos, emb_with_pos, emb_with_pos
    )

    # Plot attention weights for the first head
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_weights[0, 0].detach().numpy(), cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title("Attention Weights (First Head)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print("\nObservations:")
    print("1. As tokens pass through the encoder layers, their representations evolve.")
    print("2. The positional encoding combined with self-attention allows the model")
    print("   to contextualize each token based on the entire sequence.")
    print("3. The self-attention mechanism shows how each token attends to others,")
    print("   allowing the model to capture relationships between tokens.")


def sentiment_classification_example():
    """
    Example showing how to use the encoder for a sentiment classification task.
    """
    print("\nSentiment Classification with Transformer Encoder")
    print("-" * 50)

    # Parameters
    vocab_size = 1000
    d_model = 32
    num_heads = 4
    d_ff = 128
    num_layers = 2

    # Create a Transformer encoder
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)

    # Add a classification head
    class_head = nn.Linear(d_model, 2)  # 2 classes: positive and negative

    # Example sentences (token indices)
    # Imagine these are tokenized sentences:
    # 1. "I love this movie" (positive)
    # 2. "This film is terrible" (negative)
    positive_example = torch.tensor([[1, 2, 3, 4, 0, 0]])  # Padded to length 6
    negative_example = torch.tensor([[5, 6, 7, 8, 9, 0]])  # Padded to length 6

    sentences = torch.cat([positive_example, negative_example], dim=0)
    labels = torch.tensor([1, 0])  # 1 for positive, 0 for negative

    # Create padding mask
    mask = create_padding_mask(sentences)

    # Encode the sentences
    encoded, _ = encoder(sentences, mask)

    # For classification, we often take the representation of the first token
    # or apply a pooling operation over all tokens
    # Let's use mean pooling here

    # Create a mask for padding tokens (1 for real tokens, 0 for padding)
    padding_mask = (sentences != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Apply the mask and compute the mean
    masked_encoded = encoded * padding_mask
    seq_lengths = padding_mask.sum(dim=1)
    pooled = masked_encoded.sum(dim=1) / seq_lengths

    # Apply the classification head
    logits = class_head(pooled)
    predictions = torch.argmax(logits, dim=1)

    # Print results
    print(f"Input sentences shape: {sentences.shape}")
    print(f"Encoded representations shape: {encoded.shape}")
    print(f"Pooled representations shape: {pooled.shape}")
    print(f"Prediction logits shape: {logits.shape}")
    print(f"Predictions: {predictions.numpy()}")
    print(f"True labels: {labels.numpy()}")

    print("\nNote: This is just a toy example with random weights.")
    print("In a real application, you would train the model on labeled data.")


def exercise_custom_encoder():
    """
    Exercise: Customize and experiment with the Transformer encoder.

    Try modifying the Transformer encoder architecture and observe the effects.
    """
    print("\nExercise: Customize the Transformer Encoder")
    print("-" * 50)

    # Task: Modify the Transformer encoder to experiment with different components.
    # Here are some ideas:
    # 1. Change the number of heads
    # 2. Adjust the dimensionality of the feed-forward network
    # 3. Modify the positional encoding
    # 4. Add or remove layers

    print("In this exercise, experiment with the following modifications:")
    print("1. Increase the number of attention heads (e.g., from 4 to 8)")
    print("2. Add more encoder layers (e.g., from 2 to 4)")
    print("3. Change the feed-forward network dimensionality")
    print("4. Try a different pooling strategy for classification")

    # Example implementation of a modified encoder
    vocab_size = 1000
    d_model = 64  # Increased from 32
    num_heads = 8  # Increased from 4
    d_ff = 256  # Increased from 128
    num_layers = 4  # Increased from 2

    # Create the modified encoder
    modified_encoder = TransformerEncoder(
        vocab_size, d_model, num_heads, d_ff, num_layers
    )

    # Test input
    x = torch.randint(1, vocab_size, (2, 8))
    mask = create_padding_mask(x)

    # Forward pass
    encoded, attentions = modified_encoder(x, mask)

    print(f"\nModified encoder parameters:")
    print(f"- d_model: {d_model}")
    print(f"- num_heads: {num_heads}")
    print(f"- d_ff: {d_ff}")
    print(f"- num_layers: {num_layers}")

    print(f"\nEncoded output shape: {encoded.shape}")
    print(f"Number of attention layers: {len(attentions)}")

    # Challenge: Try to implement a version that uses learned positional embeddings
    # instead of the fixed sinusoidal encoding
    print("\nChallenge: Implement a version with learned positional embeddings")
    print("instead of the fixed sinusoidal encoding.")


if __name__ == "__main__":
    # Visualize positional encoding
    visualize_positional_encoding()

    # Run transformer encoder example
    transformer_encoder_example()

    # Analyze how representations evolve
    analyze_representations()

    # Sentiment classification example
    sentiment_classification_example()

    # Exercise
    exercise_custom_encoder()

    print("\nCongratulations! You've implemented and explored the Transformer encoder.")
    print("In the next exercise, you'll build a complete Transformer architecture")
    print("with both encoder and decoder components.")
