#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 1: BERT for Text Classification

This exercise covers:
1. Loading a pre-trained BERT model using Hugging Face Transformers
2. Fine-tuning BERT for sentiment analysis on movie reviews
3. Evaluating the model's performance
4. Exploring BERT's contextual word representations

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based
model pre-trained on a large corpus using masked language modeling and next sentence
prediction objectives. It can be fine-tuned for various NLP tasks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_and_prepare_data(tokenizer, max_length=128, batch_size=16):
    """
    Load and prepare the IMDB dataset for sentiment analysis.

    Args:
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for training

    Returns:
        Prepared datasets and dataloaders
    """
    print("Loading IMDB dataset...")

    # Load the IMDB dataset
    imdb = load_dataset("imdb")

    # Split the training set into training and validation
    # Use 5000 examples for validation
    train_dataset = imdb["train"].shuffle(seed=42)
    eval_dataset = imdb["test"].shuffle(seed=42)

    # Tokenize function for batch processing
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    print("Tokenizing datasets...")

    # Tokenize all examples
    train_encodings = train_dataset.map(tokenize_function, batched=True)
    eval_encodings = eval_dataset.map(tokenize_function, batched=True)

    # Format for PyTorch
    train_encodings.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    eval_encodings.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    print(f"Train dataset size: {len(train_encodings)}")
    print(f"Evaluation dataset size: {len(eval_encodings)}")

    return train_encodings, eval_encodings


def train_bert_classifier_with_trainer(
    train_dataset,
    eval_dataset,
    model,
    output_dir="./results",
    batch_size=16,
    num_epochs=3,
):
    """
    Train a BERT classifier using Hugging Face's Trainer.

    Args:
        train_dataset: Prepared training dataset
        eval_dataset: Prepared evaluation dataset
        model: Pre-trained BERT model
        output_dir: Directory to save results
        batch_size: Batch size for training
        num_epochs: Number of training epochs

    Returns:
        Trained model
    """
    print("\nTraining BERT classifier...")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Define compute_metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    return trainer


def evaluate_model(trainer, eval_dataset):
    """
    Evaluate the model on the test dataset.

    Args:
        trainer: Trained Trainer object
        eval_dataset: Evaluation dataset

    Returns:
        Evaluation metrics
    """
    print("\nEvaluating model on test dataset...")

    # Evaluate
    results = trainer.evaluate()

    # Get predictions
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Print evaluation metrics
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

    return results


def visualize_attention(model, tokenizer, text, layer=11):
    """
    Visualize BERT's attention patterns for a given text.

    Args:
        model: Fine-tuned BERT model
        tokenizer: BERT tokenizer
        text: Input text
        layer: Which transformer layer to visualize (default: last layer)
    """
    print(f"\nVisualizing attention for: '{text}'")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get model's attention
    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )

    # Get attention from the specified layer and last head
    attentions = outputs.attentions[layer][0]  # Last layer
    att_matrix = attentions[-1].cpu().numpy()  # Last head

    # Plot attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(att_matrix, cmap="viridis")

    # Set axis labels
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)

    plt.title(f"BERT Attention (Layer {layer+1}, Head {attentions.shape[0]})")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def explore_contextual_embeddings(model, tokenizer, sentences):
    """
    Explore how BERT's contextual embeddings change based on context.

    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        sentences: List of sentences containing the same word in different contexts
    """
    print("\nExploring contextual embeddings...")

    # Word to analyze
    target_word = "bank"  # Can be modified to analyze different words

    # Store embeddings of the target word
    word_embeddings = []
    word_positions = []

    # Process each sentence
    for i, sentence in enumerate(sentences):
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Find position of target word
        try:
            # Note: BERT's WordPiece tokenizer might split words, so we check for the first piece
            pos = tokens.index(target_word) if target_word in tokens else -1

            # If word not found, try to find the first piece (e.g., "bank" might be tokenized as "ban" + "##k")
            if pos == -1:
                for j, token in enumerate(tokens):
                    if token.startswith(
                        target_word[:3]
                    ):  # Look for the beginning of the word
                        pos = j
                        break

            if pos != -1:
                # Get hidden states
                with torch.no_grad():
                    outputs = model(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True,
                    )

                # Get embedding from last hidden state
                hidden_states = outputs.hidden_states[-1]  # Last layer
                word_embedding = hidden_states[0, pos, :].cpu().numpy()

                word_embeddings.append(word_embedding)
                word_positions.append((i, pos))

                print(f"Sentence {i+1}: Found '{target_word}' at position {pos}")
            else:
                print(f"Sentence {i+1}: '{target_word}' not found")

        except Exception as e:
            print(f"Error processing sentence {i+1}: {e}")

    # Compute cosine similarities between embeddings
    if len(word_embeddings) > 1:
        print("\nCosine similarities between contextual embeddings:")
        for i in range(len(word_embeddings)):
            for j in range(i + 1, len(word_embeddings)):
                emb1 = word_embeddings[i]
                emb2 = word_embeddings[j]

                # Compute cosine similarity
                cos_sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )

                print(
                    f"Similarity between '{target_word}' in sentences {word_positions[i][0]+1} and {word_positions[j][0]+1}: {cos_sim:.4f}"
                )

                # Interpretation
                if cos_sim > 0.9:
                    print("  Interpretation: Very similar meaning")
                elif cos_sim > 0.7:
                    print("  Interpretation: Similar meaning")
                elif cos_sim > 0.5:
                    print("  Interpretation: Somewhat related meaning")
                else:
                    print("  Interpretation: Different meanings")


def main():
    """
    Main function to run the BERT classification exercise.
    """
    print("BERT for Text Classification")
    print("-" * 50)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    print("\nLoading pre-trained BERT model and tokenizer...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification (positive/negative)
        output_attentions=True,
        output_hidden_states=True,
    )

    # Load and prepare data
    train_dataset, eval_dataset = load_and_prepare_data(tokenizer)

    # Train model
    trainer = train_bert_classifier_with_trainer(
        train_dataset,
        eval_dataset,
        model,
        output_dir="./bert-sentiment",
        num_epochs=3,  # Reduced for demo, use more epochs for better results
    )

    # Evaluate model
    results = evaluate_model(trainer, eval_dataset)

    # Example sentences for visualization
    positive_example = "This movie was absolutely fantastic. The acting was superb and the plot was engaging throughout."
    negative_example = "I hated this film. The storyline was boring and the characters were poorly developed."

    # Visualize attention
    visualize_attention(trainer.model, tokenizer, positive_example)

    # Explore contextual embeddings
    sentences = [
        "I went to the bank to deposit money.",
        "The river bank was covered with flowers.",
        "The bank denied my loan application.",
        "We sat on the bank and watched the sunset.",
    ]
    explore_contextual_embeddings(trainer.model, tokenizer, sentences)

    print("\nExercise complete!")


if __name__ == "__main__":
    main()
