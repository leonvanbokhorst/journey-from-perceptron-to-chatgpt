#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 4: Implementing the Complete Transformer Architecture

This exercise covers:
1. Implementing a complete Transformer with encoder and decoder
2. Training the model on a sequence-to-sequence task
3. Analyzing attention patterns between encoder and decoder

The Transformer architecture represents a breakthrough in sequence-to-sequence
modeling, enabling efficient processing of sequences without recurrence or convolution.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
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
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """
    Layer Normalization.
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
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
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
        # Self-Attention sub-layer
        attn_output, self_attn_weights = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)  # Residual connection and layer norm

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)  # Residual connection and layer norm

        return x, self_attn_weights


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Each decoder layer consists of:
    1. Masked multi-head self-attention (prevents attending to future positions)
    2. Multi-head cross-attention (attends to encoder outputs)
    3. Position-wise feed-forward network

    Each sub-layer has a residual connection and is followed by layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input tensor
            encoder_output: Output from the encoder
            src_mask: Mask for padding in the encoder output
            tgt_mask: Mask to prevent attending to future positions in the decoder input

        Returns:
            Decoder layer output and attention weights
        """
        # Self-Attention sub-layer
        attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # Cross-Attention sub-layer
        attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        attn_output = self.dropout2(attn_output)
        x = self.norm2(x + attn_output)

        # Feed-Forward sub-layer
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.norm3(x + ff_output)

        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence tasks.

    The model consists of an encoder and a decoder. The encoder processes the input sequence
    and the decoder generates the output sequence given the encoder's output.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1,
        max_len=5000,
    ):
        super(Transformer, self).__init__()

        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final linear layer and softmax for output
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)

        # Normalization layers
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)

        # Model dimensions
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            src_mask: Optional mask for padding in the source

        Returns:
            Encoder output and attention weights
        """
        # Convert token indices to embeddings and scale
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply dropout
        x = self.dropout(x)

        # Process through encoder layers
        encoder_attentions = []
        for layer in self.encoder_layers:
            x, attention = layer(x, src_mask)
            encoder_attentions.append(attention)

        # Apply final layer normalization
        x = self.encoder_norm(x)

        return x, encoder_attentions

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode the target sequence given the encoder output.

        Args:
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            encoder_output: Output from the encoder
            src_mask: Mask for padding in the source
            tgt_mask: Mask to prevent attending to future positions in the target

        Returns:
            Decoder output and attention weights
        """
        # Convert token indices to embeddings and scale
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply dropout
        x = self.dropout(x)

        # Process through decoder layers
        self_attentions = []
        cross_attentions = []
        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)

        # Apply final layer normalization
        x = self.decoder_norm(x)

        # Apply final linear projection
        output = self.final_layer(x)

        return output, self_attentions, cross_attentions

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the complete Transformer.

        Args:
            src: Source sequence tensor of shape (batch_size, src_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_len)
            src_mask: Mask for padding in the source
            tgt_mask: Mask to prevent attending to future positions in the target

        Returns:
            Output logits and attention weights
        """
        # Encode
        encoder_output, encoder_attentions = self.encode(src, src_mask)

        # Decode
        output, self_attentions, cross_attentions = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )

        return output, {
            "encoder_attentions": encoder_attentions,
            "decoder_self_attentions": self_attentions,
            "decoder_cross_attentions": cross_attentions,
        }


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


def create_look_ahead_mask(seq_len):
    """
    Create a mask to prevent attention to future tokens.

    Args:
        seq_len: Length of the sequence

    Returns:
        Mask tensor of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
    return mask


def create_masks(src, tgt, pad_idx=0):
    """
    Create masks for transformer training.

    Args:
        src: Source sequence tensor of shape (batch_size, src_len)
        tgt: Target sequence tensor of shape (batch_size, tgt_len)
        pad_idx: Padding token index

    Returns:
        Source mask and target mask
    """
    # Source mask (for padding)
    src_mask = create_padding_mask(src, pad_idx)

    # Target mask (combines padding mask and look-ahead mask)
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len)

    # Combine the masks
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

    return src_mask, tgt_mask


def visualize_attention(
    attention_weights, src_tokens=None, tgt_tokens=None, title=None
):
    """
    Visualize attention weights.

    Args:
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        src_tokens: Optional list of source tokens
        tgt_tokens: Optional list of target tokens
        title: Optional title for the plot
    """
    batch_size, num_heads, seq_len_q, seq_len_k = attention_weights.shape

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4))

    if num_heads == 1:
        axes = [axes]  # Make it iterable

    # Plot each head's attention pattern
    for h, ax in enumerate(axes):
        weights = attention_weights[0, h].detach().numpy()

        # Create heatmap
        im = ax.imshow(weights, cmap="viridis")
        plt.colorbar(im, ax=ax)

        # Set labels
        if src_tokens and tgt_tokens:
            ax.set_xticks(range(len(src_tokens)))
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_xticklabels(src_tokens, rotation=90)
            ax.set_yticklabels(tgt_tokens)
        elif src_tokens:  # Self-attention
            ax.set_xticks(range(len(src_tokens)))
            ax.set_yticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=90)
            ax.set_yticklabels(src_tokens)

        if title:
            ax.set_title(f"Head {h+1}")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def toy_sequence_translation():
    """
    Example demonstrating a toy sequence translation task.

    This example uses a small synthetic task to demonstrate the Transformer:
    Translating a sequence of numbers to their sorted version.
    """
    print("Toy Sequence Translation Example")
    print("-" * 50)

    # Parameters
    vocab_size = 20  # Tokens 0-19 (0 is padding, 1 is start/end token)
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2

    # Create a smaller transformer for this toy task
    transformer = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )

    # Define a simple example
    # Source: A scrambled sequence of numbers
    # Target: The same numbers sorted in ascending order
    src_seq = torch.tensor([[5, 3, 8, 2, 9, 0, 0]])  # Last two are padding
    # For the target, we prepend the "start token" (1) during training
    tgt_seq_in = torch.tensor(
        [[1, 2, 3, 5, 8, 9, 0]]
    )  # Input to decoder during training
    # We expect the model to predict these tokens (shifted by one position)
    tgt_seq_out = torch.tensor([[2, 3, 5, 8, 9, 0, 0]])  # Expected output

    # Create masks
    src_mask, tgt_mask = create_masks(src_seq, tgt_seq_in)

    # Forward pass
    output, attentions = transformer(src_seq, tgt_seq_in, src_mask, tgt_mask)

    # Output has shape (batch_size, tgt_len, vocab_size)
    # Convert to predictions by taking argmax
    predictions = torch.argmax(output, dim=-1)

    # Print results
    print("Source sequence:", src_seq[0].tolist())
    print("Target sequence (input):", tgt_seq_in[0].tolist())
    print("Expected output:", tgt_seq_out[0].tolist())
    print("Predicted output:", predictions[0].tolist())

    # Visualization of attention
    print("\nVisualizing attention patterns:")

    # Encoder self-attention
    encoder_attn = attentions["encoder_attentions"][0]  # First encoder layer
    src_tokens = [str(t.item()) for t in src_seq[0] if t.item() != 0]
    visualize_attention(encoder_attn, src_tokens, title="Encoder Self-Attention")

    # Decoder self-attention
    decoder_self_attn = attentions["decoder_self_attentions"][0]  # First decoder layer
    tgt_tokens = [str(t.item()) for t in tgt_seq_in[0] if t.item() != 0]
    visualize_attention(decoder_self_attn, tgt_tokens, title="Decoder Self-Attention")

    # Cross-attention
    cross_attn = attentions["decoder_cross_attentions"][0]  # First decoder layer
    visualize_attention(cross_attn, src_tokens, tgt_tokens, title="Cross-Attention")

    print("\nNote: The Transformer needs to be trained before it can sort correctly.")
    print("This is just a demonstration of the forward pass.")


def train_transformer_toy_task(batch_size=64, num_batches=100):
    """
    Train the Transformer on a toy task: sorting sequences of numbers.

    Args:
        batch_size: Batch size for training
        num_batches: Number of batches to train for
    """
    print("\nTraining Transformer on Sorting Task")
    print("-" * 50)

    # Parameters
    vocab_size = 20
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2

    # Create a transformer
    transformer = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )  # Ignore padding in loss calculation

    # Training loop
    transformer.train()
    for batch in range(num_batches):
        # Generate random sequences for sorting
        max_len = 8
        seq_len = torch.randint(3, max_len, (batch_size,))
        src_batch = []
        tgt_batch_in = []
        tgt_batch_out = []

        for i in range(batch_size):
            # Generate a random sequence of length seq_len[i]
            length = seq_len[i].item()
            # Use numbers 2-19 (reserve 0 for padding, 1 for start token)
            seq = torch.randint(2, vocab_size, (length,))

            # Sort the sequence
            sorted_seq, _ = torch.sort(seq)

            # Create source sequence (padded to max_len)
            padded_src = torch.zeros(max_len, dtype=torch.long)
            padded_src[:length] = seq
            src_batch.append(padded_src)

            # Create target sequence input (prepend start token, pad to max_len)
            padded_tgt_in = torch.zeros(max_len, dtype=torch.long)
            padded_tgt_in[0] = 1  # Start token
            padded_tgt_in[1 : length + 1] = sorted_seq
            tgt_batch_in.append(padded_tgt_in)

            # Create target sequence output (shifted by one, pad to max_len)
            padded_tgt_out = torch.zeros(max_len, dtype=torch.long)
            padded_tgt_out[:length] = sorted_seq
            tgt_batch_out.append(padded_tgt_out)

        # Stack into batches
        src_batch = torch.stack(src_batch)
        tgt_batch_in = torch.stack(tgt_batch_in)
        tgt_batch_out = torch.stack(tgt_batch_out)

        # Create masks
        src_mask, tgt_mask = create_masks(src_batch, tgt_batch_in)

        # Forward pass
        optimizer.zero_grad()
        output, _ = transformer(src_batch, tgt_batch_in, src_mask, tgt_mask)

        # Compute loss
        loss = criterion(
            output.contiguous().view(-1, vocab_size),
            tgt_batch_out.contiguous().view(-1),
        )

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print progress
        if (batch + 1) % 10 == 0:
            print(f"Batch {batch + 1}/{num_batches}, Loss: {loss.item():.4f}")

    print("\nTraining complete!")

    # Test the model on a new example
    print("\nTesting the trained model:")
    transformer.eval()

    # Create a test sequence
    test_src = torch.tensor([[7, 4, 9, 3, 6, 0, 0, 0]])  # Padded to max_len

    # Greedy decoding
    max_len = test_src.size(1)
    src_mask = create_padding_mask(test_src)

    # Encode the source sequence
    encoder_output, _ = transformer.encode(test_src, src_mask)

    # Initialize the target sequence with the start token
    tgt = torch.zeros(1, max_len, dtype=torch.long)
    tgt[0, 0] = 1  # Start token

    # Generate the output sequence
    for i in range(1, max_len):
        # Create mask for the target sequence
        tgt_mask = create_look_ahead_mask(i)

        # Decode the target sequence so far
        output, _, _ = transformer.decode(
            tgt[:, :i], encoder_output, src_mask, tgt_mask
        )

        # Get the next token prediction
        next_token = torch.argmax(output[:, -1], dim=-1)

        # Add to the target sequence
        tgt[0, i] = next_token

        # Stop if we predict padding or end of sequence
        if next_token == 0:
            break

    # Print results
    print("Source sequence:", [t.item() for t in test_src[0] if t.item() != 0])
    print(
        "Predicted (sorted) sequence:",
        [t.item() for t in tgt[0] if t.item() not in [0, 1]],
    )

    # Get the correct sorted sequence for comparison
    test_src_no_pad = test_src[0][test_src[0] != 0]
    sorted_src, _ = torch.sort(test_src_no_pad)
    print("Expected (sorted) sequence:", sorted_src.tolist())


def exercise_customization():
    """
    Exercise: Customize the Transformer for a different task.

    In this exercise, you'll modify the Transformer for a different sequence-to-sequence task.
    """
    print("\nExercise: Customize the Transformer")
    print("-" * 50)

    print("Ideas for customization:")
    print("1. Implement a different positional encoding scheme")
    print("2. Modify the architecture for a classification task")
    print("3. Implement a simplified version with fewer layers")
    print("4. Create a task-specific Transformer (e.g., for time series prediction)")

    # Example: A simplified Transformer for classification
    print("\nExample: Simplified Transformer for Classification")

    class TransformerClassifier(nn.Module):
        """
        A simplified Transformer for sequence classification.
        Uses only the encoder part followed by a classification head.
        """

        def __init__(
            self,
            vocab_size,
            num_classes,
            d_model=64,
            num_heads=4,
            d_ff=256,
            num_layers=2,
            dropout=0.1,
        ):
            super(TransformerClassifier, self).__init__()

            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model)
            self.dropout = nn.Dropout(dropout)

            # Encoder layers
            self.encoder_layers = nn.ModuleList(
                [
                    EncoderLayer(d_model, num_heads, d_ff, dropout)
                    for _ in range(num_layers)
                ]
            )

            self.encoder_norm = LayerNorm(d_model)

            # Classification head
            self.classifier = nn.Linear(d_model, num_classes)

            # Model dimensions
            self.d_model = d_model

        def forward(self, x, mask=None):
            # Embedding and positional encoding
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.positional_encoding(x)
            x = self.dropout(x)

            # Encoder layers
            attentions = []
            for layer in self.encoder_layers:
                x, attention = layer(x, mask)
                attentions.append(attention)

            # Apply final layer normalization
            x = self.encoder_norm(x)

            # Global pooling (mean of non-padding tokens)
            # Create a mask for padding tokens (1 for real tokens, 0 for padding)
            padding_mask = (x != 0).float().unsqueeze(-1)
            masked_x = x * padding_mask
            pooled = masked_x.sum(dim=1) / padding_mask.sum(dim=1)

            # Classification
            logits = self.classifier(pooled)

            return logits, attentions

    # Create a toy dataset
    vocab_size = 100
    num_classes = 5
    batch_size = 8
    seq_len = 10

    # Random sequences
    sequences = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Random labels
    labels = torch.randint(0, num_classes, (batch_size,))

    # Create the model
    classifier = TransformerClassifier(vocab_size, num_classes)

    # Forward pass
    mask = create_padding_mask(sequences)
    logits, attentions = classifier(sequences, mask)

    # Print results
    print(f"Input sequences shape: {sequences.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Compute predictions
    predictions = torch.argmax(logits, dim=-1)

    print(f"Predictions: {predictions.tolist()}")
    print(f"True labels: {labels.tolist()}")

    print("\nYour challenge: Modify one aspect of the Transformer architecture")
    print("and experiment with how it affects the model's behavior or performance.")


if __name__ == "__main__":
    # Show the toy sequence translation example
    toy_sequence_translation()

    # Train the Transformer on a toy task
    train_transformer_toy_task(batch_size=32, num_batches=50)

    # Exercise on customizing the Transformer
    exercise_customization()

    print(
        "\nCongratulations! You've implemented and explored the complete Transformer architecture."
    )
    print(
        "You now understand the fundamental components of modern language models like BERT and GPT."
    )
    print(
        "In the next module, we'll explore how these models are pre-trained and fine-tuned for specific tasks."
    )
