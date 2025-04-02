"""
Sentiment Analysis with Bidirectional LSTM/GRU

This exercise implements bidirectional LSTM and GRU networks for sentiment
analysis on movie reviews, demonstrating their power in NLP tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import string
from collections import Counter
from typing import Tuple, List, Dict, Optional, Any


class MovieReviewDataset(Dataset):
    """
    Dataset for loading and processing movie reviews.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_length: int = 200,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text reviews
            labels: List of sentiment labels (0 or 1)
            vocab: Dictionary mapping words to indices
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a review and its label by index.

        Args:
            idx: Index

        Returns:
            Tuple of (text_tensor, label_tensor)
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to sequence of word indices
        words = text.split()
        indices = [
            self.vocab.get(word, self.vocab["<UNK>"])
            for word in words[: self.max_length]
        ]

        # Pad sequence
        if len(indices) < self.max_length:
            indices += [self.vocab["<PAD>"]] * (self.max_length - len(indices))

        # Convert to tensors
        text_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return text_tensor, label_tensor


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model for sentiment analysis.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        dropout: float = 0.5,
    ):
        """
        Initialize the model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            output_dim: Dimension of output (number of classes)
            n_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(BidirectionalLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text: Input tensor of token indices (batch_size, seq_len)

        Returns:
            Output tensor of class logits (batch_size, output_dim)
        """
        # text = [batch size, seq len]
        embedded = self.embedding(text)
        # embedded = [batch size, seq len, embedding dim]

        output, (hidden, cell) = self.lstm(embedded)
        # output = [batch size, seq len, hidden dim * 2]
        # hidden = [n layers * 2, batch size, hidden dim]

        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hidden dim * 2]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU model for sentiment analysis.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        dropout: float = 0.5,
    ):
        """
        Initialize the model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            output_dim: Dimension of output (number of classes)
            n_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(BidirectionalGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text: Input tensor of token indices (batch_size, seq_len)

        Returns:
            Output tensor of class logits (batch_size, output_dim)
        """
        # text = [batch size, seq len]
        embedded = self.embedding(text)
        # embedded = [batch size, seq len, embedding dim]

        output, hidden = self.gru(embedded)
        # output = [batch size, seq len, hidden dim * 2]
        # hidden = [n layers * 2, batch size, hidden dim]

        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hidden dim * 2]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


def preprocess_text(text: str) -> str:
    """
    Preprocess a text by lowercasing, removing punctuation, and extra whitespace.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = "".join([c for c in text if c not in string.punctuation])

    # Remove extra whitespace
    text = re.sub("\s+", " ", text).strip()

    return text


def build_vocab(
    texts: List[str], min_freq: int = 2, max_size: int = 25000
) -> Dict[str, int]:
    """
    Build a vocabulary from texts.

    Args:
        texts: List of texts
        min_freq: Minimum word frequency to include in vocab
        max_size: Maximum vocabulary size

    Returns:
        Vocabulary dictionary mapping words to indices
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # Filter by frequency and limit size
    common_words = [
        word for word, count in word_counts.most_common(max_size) if count >= min_freq
    ]

    # Create vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(common_words, start=2):
        vocab[word] = i

    return vocab


def fetch_data() -> Tuple[List[str], List[int]]:
    """
    Fetch movie review data. If the IMDB dataset is not available,
    generate synthetic data for demonstration.

    Returns:
        Tuple of (texts, labels)
    """
    try:
        # Try to import the IMDB dataset from torchtext
        from torchtext.datasets import IMDB

        print("Using the IMDB dataset from torchtext...")

        # Get train data (in practice, we'd also process validation data)
        train_iter = IMDB(split="train")

        texts = []
        labels = []

        for label, text in train_iter:
            texts.append(preprocess_text(text))
            labels.append(1 if label == "pos" else 0)

        return texts, labels

    except:
        print("IMDB dataset not available. Generating synthetic data...")

        # Generate synthetic data for demonstration
        n_samples = 1000
        texts = []
        labels = []

        # Positive sentiment words
        positive_words = [
            "good",
            "great",
            "excellent",
            "wonderful",
            "amazing",
            "enjoyable",
            "liked",
            "fantastic",
            "awesome",
            "love",
            "best",
            "favorite",
            "recommend",
            "happy",
        ]

        # Negative sentiment words
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disappointing",
            "disliked",
            "poor",
            "worst",
            "waste",
            "boring",
            "hate",
            "avoid",
            "failed",
            "sad",
        ]

        # Neutral words for padding sentences
        neutral_words = [
            "movie",
            "film",
            "watch",
            "scene",
            "actor",
            "actress",
            "director",
            "character",
            "story",
            "plot",
            "cinema",
            "theater",
            "show",
            "production",
        ]

        # Generate synthetic reviews
        for i in range(n_samples):
            # Decide sentiment (50/50 split)
            sentiment = i % 2

            # Choose number of words
            n_words = np.random.randint(10, 100)

            if sentiment == 1:  # Positive
                # Generate a review with mostly positive sentiment words
                sentiment_words = positive_words
                opposite_words = negative_words
            else:  # Negative
                # Generate a review with mostly negative sentiment words
                sentiment_words = negative_words
                opposite_words = positive_words

            # Generate text
            words = []
            for j in range(n_words):
                r = np.random.random()
                if r < 0.4:  # 40% chance of sentiment word
                    words.append(np.random.choice(sentiment_words))
                elif r < 0.5:  # 10% chance of opposite sentiment word
                    words.append(np.random.choice(opposite_words))
                else:  # 50% chance of neutral word
                    words.append(np.random.choice(neutral_words))

            text = " ".join(words)
            texts.append(text)
            labels.append(sentiment)

        return texts, labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.001,
) -> Tuple[List[float], List[float]]:
    """
    Train a model.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Tuple of (train_losses, valid_losses)
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training and validation losses
    train_losses = []
    valid_losses = []

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(texts)

            # Calculate loss
            loss = criterion(predictions, labels)

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Update parameters
            optimizer.step()

            # Track loss and accuracy
            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        epoch_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in valid_loader:
                texts, labels = batch
                texts, labels = texts.to(device), labels.to(device)

                # Forward pass
                predictions = model(texts)

                # Calculate loss
                loss = criterion(predictions, labels)

                # Track loss and accuracy
                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(predictions, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        valid_loss = epoch_loss / len(valid_loader)
        valid_accuracy = correct / total
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(
            f"  Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}"
        )

    return train_losses, valid_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
    """
    Evaluate a model on test data.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data

    Returns:
        Dictionary of evaluation metrics
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Track metrics
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            # Forward pass
            predictions = model(texts)

            # Calculate accuracy
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store predictions and labels for further analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = correct / total
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, output_dict=True)

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def visualize_results(
    lstm_results: Dict[str, Any],
    gru_results: Dict[str, Any],
    lstm_history: Tuple[List[float], List[float]],
    gru_history: Tuple[List[float], List[float]],
) -> None:
    """
    Visualize training history and evaluation results.

    Args:
        lstm_results: LSTM evaluation results
        gru_results: GRU evaluation results
        lstm_history: LSTM training history (train_losses, valid_losses)
        gru_history: GRU training history (train_losses, valid_losses)
    """
    # Plot training and validation losses
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lstm_history[0], label="LSTM Train Loss")
    plt.plot(lstm_history[1], label="LSTM Validation Loss")
    plt.plot(gru_history[0], label="GRU Train Loss")
    plt.plot(gru_history[1], label="GRU Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot confusion matrices
    plt.subplot(1, 2, 2)

    lstm_cm = lstm_results["confusion_matrix"]
    gru_cm = gru_results["confusion_matrix"]

    # Combine confusion matrices for side-by-side comparison
    cm = np.zeros((2, 4))
    cm[0, 0:2] = lstm_cm[0]
    cm[1, 0:2] = lstm_cm[1]
    cm[0, 2:4] = gru_cm[0]
    cm[1, 2:4] = gru_cm[1]

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrices: LSTM (left) vs GRU (right)")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(np.arange(4), ["Neg", "Pos", "Neg", "Pos"])
    plt.yticks(tick_marks, ["Neg", "Pos"])

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(4):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()

    # Print metrics comparison
    print("\nMetrics Comparison:")
    print(f"LSTM Accuracy: {lstm_results['accuracy']:.4f}")
    print(f"GRU Accuracy: {gru_results['accuracy']:.4f}")

    print("\nLSTM Classification Report:")
    lstm_report = lstm_results["classification_report"]
    print(f"  Negative Precision: {lstm_report['0']['precision']:.4f}")
    print(f"  Positive Precision: {lstm_report['1']['precision']:.4f}")
    print(f"  Negative Recall: {lstm_report['0']['recall']:.4f}")
    print(f"  Positive Recall: {lstm_report['1']['recall']:.4f}")

    print("\nGRU Classification Report:")
    gru_report = gru_results["classification_report"]
    print(f"  Negative Precision: {gru_report['0']['precision']:.4f}")
    print(f"  Positive Precision: {gru_report['1']['precision']:.4f}")
    print(f"  Negative Recall: {gru_report['0']['recall']:.4f}")
    print(f"  Positive Recall: {gru_report['1']['recall']:.4f}")


def compare_attention_mechanism() -> None:
    """
    Explain and demonstrate the attention mechanism as an extension to bidirectional RNNs.
    This is a theoretical explanation with a simple demonstration.
    """
    print("\nAttention Mechanism Demonstration:")
    print("==================================")
    print(
        "Attention allows the model to focus on different parts of the input sequence"
    )
    print(
        "when making predictions, similar to how humans pay more attention to certain"
    )
    print("words when understanding sentiment.")

    # Example sentence and attention weights
    sentence = "The movie was not good but the actors did a fantastic job"
    words = sentence.split()

    # Simulated attention weights (would normally come from a model)
    attention_weights = np.array(
        [0.02, 0.03, 0.15, 0.20, 0.25, 0.03, 0.02, 0.05, 0.05, 0.20]
    )

    # Plot the attention weights
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(words)), attention_weights, align="center", alpha=0.7)
    plt.xticks(range(len(words)), words, rotation=45)
    plt.title("Example of Attention Weights on a Sentence")
    plt.xlabel("Words")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.show()

    print("\nNote how the model pays more attention to sentiment-bearing words")
    print("like 'not', 'good', and 'fantastic' when determining overall sentiment.")
    print(
        "This allows the model to better handle complex sentiment patterns like negation."
    )


def main() -> None:
    """
    Main function to run the sentiment analysis exercise.
    """
    print("Sentiment Analysis with Bidirectional LSTM/GRU")
    print("=============================================")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fetch data
    texts, labels = fetch_data()
    print(f"Loaded {len(texts)} reviews")

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        train_texts, train_labels, test_size=0.25, random_state=42
    )  # 0.25 x 0.8 = 0.2

    print(f"Training set: {len(train_texts)} reviews")
    print(f"Validation set: {len(valid_texts)} reviews")
    print(f"Test set: {len(test_texts)} reviews")

    # Build vocabulary
    vocab = build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Create datasets
    train_dataset = MovieReviewDataset(train_texts, train_labels, vocab)
    valid_dataset = MovieReviewDataset(valid_texts, valid_labels, vocab)
    test_dataset = MovieReviewDataset(test_texts, test_labels, vocab)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # Binary sentiment (positive/negative)
    n_layers = 2
    dropout = 0.5

    # Initialize models
    lstm_model = BidirectionalLSTM(
        vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout
    )
    gru_model = BidirectionalGRU(
        vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout
    )

    print("\nTraining Bidirectional LSTM...")
    lstm_history = train_model(lstm_model, train_loader, valid_loader, epochs=5)

    print("\nTraining Bidirectional GRU...")
    gru_history = train_model(gru_model, train_loader, valid_loader, epochs=5)

    # Evaluate models
    print("\nEvaluating Bidirectional LSTM...")
    lstm_results = evaluate_model(lstm_model, test_loader)

    print("\nEvaluating Bidirectional GRU...")
    gru_results = evaluate_model(gru_model, test_loader)

    # Visualize results
    visualize_results(lstm_results, gru_results, lstm_history, gru_history)

    # Demonstrate attention mechanism
    compare_attention_mechanism()

    print("\nExercise completed!")


if __name__ == "__main__":
    main()
