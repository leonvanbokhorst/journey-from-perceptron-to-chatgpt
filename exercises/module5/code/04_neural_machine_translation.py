#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise: Neural Machine Translation with Attention

This exercise demonstrates a practical application of sequence-to-sequence models
with attention for neural machine translation on a small English-French dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import re
import time
from typing import List, Dict, Tuple, Optional
import pickle
import os

# Ensure reproducibility
np.random.seed(42)
random.seed(42)


def preprocess_sentence(sentence: str) -> str:
    """Clean and normalize a sentence for translation."""
    # Convert to lowercase and remove special characters
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()

    # Add start and end tokens
    return "<start> " + sentence + " <end>"


def create_dataset(path: str, num_examples: int) -> Tuple[List[str], List[str]]:
    """Create a dataset from a text file with English-French sentence pairs."""
    lines = open(path, encoding="UTF-8").read().strip().split("\n")

    # Extract sentence pairs and clean them
    sentence_pairs = [
        [preprocess_sentence(s) for s in l.split("\t")]
        for l in lines[: min(num_examples, len(lines))]
    ]

    return zip(*sentence_pairs)


def tokenize(
    texts: List[str],
) -> Tuple[Dict[str, int], Dict[int, str], List[List[int]]]:
    """Create vocabulary and convert texts to token indices."""
    # Build vocabulary
    vocab = {}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Reserve 0 for padding

    # Add padding token
    vocab["<pad>"] = 0

    # Create reverse vocabulary
    idx_to_word = {idx: word for word, idx in vocab.items()}

    # Convert texts to sequences of indices
    sequences = []
    for text in texts:
        sequence = [vocab[word] for word in text.split()]
        sequences.append(sequence)

    return vocab, idx_to_word, sequences


def pad_sequences(
    sequences: List[List[int]], max_len: Optional[int] = None
) -> np.ndarray:
    """Pad sequences to the same length."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq

    return padded_sequences


def load_dataset(path: str, num_examples: int = 10000) -> Dict[str, Any]:
    """Load and prepare the dataset for translation."""
    # Check if preprocessed data exists
    cache_path = f"{path}_preprocessed_{num_examples}.pkl"
    if os.path.exists(cache_path):
        print(f"Loading preprocessed data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Load and preprocess data
    print(f"Preprocessing data from {path}")
    input_texts, target_texts = create_dataset(path, num_examples)

    # Create vocabularies
    input_vocab, input_idx_to_word, input_sequences = tokenize(input_texts)
    target_vocab, target_idx_to_word, target_sequences = tokenize(target_texts)

    # Pad sequences
    input_padded = pad_sequences(input_sequences)
    target_padded = pad_sequences(target_sequences)

    # Create dataset dictionary
    dataset = {
        "input_texts": input_texts,
        "target_texts": target_texts,
        "input_vocab": input_vocab,
        "target_vocab": target_vocab,
        "input_idx_to_word": input_idx_to_word,
        "target_idx_to_word": target_idx_to_word,
        "input_sequences": input_padded,
        "target_sequences": target_padded,
    }

    # Cache the preprocessed data
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def create_demo_translation_dataset() -> Dict[str, Any]:
    """Create a small demo translation dataset."""
    # Sample English-French sentence pairs
    en_fr_pairs = [
        ("Hello, how are you?", "Bonjour, comment allez-vous?"),
        ("I love programming.", "J'aime programmer."),
        ("What's your name?", "Comment t'appelles-tu?"),
        ("The weather is nice today.", "Le temps est beau aujourd'hui."),
        ("I'm learning French.", "J'apprends le français."),
        ("Where is the library?", "Où est la bibliothèque?"),
        ("Do you speak English?", "Parlez-vous anglais?"),
        ("I'm hungry.", "J'ai faim."),
        ("How much does this cost?", "Combien ça coûte?"),
        ("See you tomorrow!", "À demain!"),
    ]

    # Preprocess sentences
    input_texts = [preprocess_sentence(pair[0]) for pair in en_fr_pairs]
    target_texts = [preprocess_sentence(pair[1]) for pair in en_fr_pairs]

    # Create vocabularies
    input_vocab, input_idx_to_word, input_sequences = tokenize(input_texts)
    target_vocab, target_idx_to_word, target_sequences = tokenize(target_texts)

    # Pad sequences
    input_padded = pad_sequences(input_sequences)
    target_padded = pad_sequences(target_sequences)

    # Create dataset dictionary
    dataset = {
        "input_texts": input_texts,
        "target_texts": target_texts,
        "input_vocab": input_vocab,
        "target_vocab": target_vocab,
        "input_idx_to_word": input_idx_to_word,
        "target_idx_to_word": target_idx_to_word,
        "input_sequences": input_padded,
        "target_sequences": target_padded,
    }

    return dataset


def visualize_attention(
    attention: np.ndarray, input_text: str, predicted_text: str
) -> None:
    """Visualize attention weights between input and output sentences."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot attention weights
    cax = ax.matshow(attention, cmap="viridis")
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticklabels([""] + input_text.split(), rotation=90)
    ax.set_yticklabels([""] + predicted_text.split())

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.title("Attention Visualization")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to demonstrate NMT with attention."""
    print("Neural Machine Translation with Attention")
    print("=========================================")

    # Create demo dataset
    print("Creating demo translation dataset...")
    dataset = create_demo_translation_dataset()

    print(f"Input vocabulary size: {len(dataset['input_vocab'])}")
    print(f"Target vocabulary size: {len(dataset['target_vocab'])}")

    # Show example sentence pairs
    print("\nExample sentence pairs:")
    for i in range(3):
        print(f"English: {dataset['input_texts'][i]}")
        print(f"French: {dataset['target_texts'][i]}")
        print()

    # Pretend to train a model (in a real implementation, we would train a PyTorch model here)
    print(
        "In a real implementation, we would now train a sequence-to-sequence model with attention."
    )
    print("For this demo, we'll skip ahead to visualization.")

    # Generate a simulated attention matrix for visualization
    input_sentence = "Hello, how are you?"
    output_sentence = "Bonjour, comment allez-vous?"

    input_tokens = preprocess_sentence(input_sentence).split()
    output_tokens = preprocess_sentence(output_sentence).split()

    # Create a simulated attention matrix
    attention_matrix = np.zeros((len(output_tokens), len(input_tokens)))

    # Simulate diagonal-dominant attention (typical pattern)
    for i in range(len(output_tokens)):
        for j in range(len(input_tokens)):
            # More attention near the diagonal, with some noise
            attention_matrix[i, j] = np.exp(
                -0.5 * ((i / len(output_tokens) - j / len(input_tokens)) ** 2)
            )

    # Normalize
    attention_matrix /= attention_matrix.sum(axis=1, keepdims=True)

    # Visualize the attention
    print("\nVisualizing attention in translation...")
    visualize_attention(
        attention_matrix, " ".join(input_tokens), " ".join(output_tokens)
    )

    print("\nExercise completed!")
    print("\nTasks for the reader:")
    print("1. Implement a full sequence-to-sequence model with attention in PyTorch.")
    print("2. Train the model on a larger dataset like Multi30k.")
    print("3. Experiment with different attention mechanisms.")
    print("4. Compare the performance against a transformer-based translation model.")


if __name__ == "__main__":
    main()
