#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 3: Fine-tuning GPT-2

This exercise covers:
1. Fine-tuning a GPT-2 model on a specialized text corpus
2. Observing how the model adapts to a specific style or domain
3. Generating text with the fine-tuned model and comparing with the base model
4. Understanding the practical considerations of fine-tuning large models

Fine-tuning allows us to adapt a pre-trained model to a specific domain or task,
leveraging its general language capabilities while specializing it for our needs.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    AdamW,
    get_linear_schedule_with_warmup,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
import logging
import textwrap
from datasets import load_dataset
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Configure basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def prepare_dataset(
    dataset_name="tiny_shakespeare", use_subset=True, subset_size=1000, cache_dir=None
):
    """
    Prepare a dataset for fine-tuning GPT-2.

    Args:
        dataset_name: Name of the dataset (e.g., "tiny_shakespeare")
        use_subset: Whether to use a subset of the data for faster fine-tuning
        subset_size: Number of examples to use if use_subset is True
        cache_dir: Directory to store cached data

    Returns:
        The prepared dataset
    """
    logger.info(f"Preparing dataset: {dataset_name}")

    if dataset_name == "tiny_shakespeare":
        # A small Shakespeare dataset for fine-tuning
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        # Create the dataset directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Download the Shakespeare data
        import requests

        logger.info("Downloading Shakespeare data...")
        r = requests.get(url)
        with open("data/shakespeare.txt", "w") as f:
            f.write(r.text)

        # Read the data
        with open("data/shakespeare.txt", "r") as f:
            data = f.read()

        # Create a text dataset (one example per line)
        import nltk

        sentences = nltk.sent_tokenize(data)

        # Use a subset if needed
        if use_subset:
            sentences = sentences[:subset_size]

        # Write to a file
        with open("data/shakespeare_lines.txt", "w") as f:
            for sentence in sentences:
                # Remove any newlines in the sentence and add one at the end
                f.write(sentence.replace("\n", " ") + "\n")

        dataset_path = "data/shakespeare_lines.txt"

    elif dataset_name == "wikitext":
        # Load wikitext dataset from Hugging Face
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)

        # Filter out empty lines
        texts = [text for text in wikitext["train"]["text"] if text.strip()]

        # Use a subset if needed
        if use_subset:
            texts = texts[:subset_size]

        # Write to a file
        with open("data/wikitext_subset.txt", "w") as f:
            for text in texts:
                f.write(text + "\n")

        dataset_path = "data/wikitext_subset.txt"

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    logger.info(f"Dataset prepared at {dataset_path}")
    return dataset_path


def prepare_custom_dataset(texts, output_path="data/custom_dataset.txt"):
    """
    Prepare a custom dataset from a list of texts.

    Args:
        texts: List of text strings to use as the dataset
        output_path: Path to save the dataset

    Returns:
        Path to the prepared dataset
    """
    logger.info("Preparing custom dataset")

    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the texts to a file (one per line)
    with open(output_path, "w") as f:
        for text in texts:
            # Clean the text: remove newlines and ensure it ends with a newline
            f.write(text.replace("\n", " ").strip() + "\n")

    logger.info(f"Custom dataset prepared at {output_path}")
    return output_path


def load_and_tokenize_dataset(file_path, tokenizer, block_size=128):
    """
    Load and tokenize dataset for language modeling.

    Args:
        file_path: Path to the text file
        tokenizer: GPT-2 tokenizer
        block_size: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    logger.info(f"Loading and tokenizing dataset from {file_path}")

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

    logger.info(f"Dataset loaded with {len(dataset)} examples")
    return dataset


def finetune_gpt2(
    model_name="gpt2",
    dataset_path=None,
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    block_size=128,
    save_steps=1000,
    save_total_limit=2,
):
    """
    Fine-tune GPT-2 on a text dataset.

    Args:
        model_name: The GPT-2 model to fine-tune
        dataset_path: Path to the dataset
        output_dir: Directory to save the model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        block_size: Maximum sequence length
        save_steps: Save model every this many steps
        save_total_limit: Maximum number of saved models

    Returns:
        The fine-tuned model and tokenizer
    """
    logger.info(f"Fine-tuning {model_name} on {dataset_path}")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    train_dataset = load_and_tokenize_dataset(
        file_path=dataset_path,
        tokenizer=tokenizer,
        block_size=block_size,
    )

    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal language modeling (CLM), not masked LM
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=True,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def compare_models(
    original_model_name="gpt2",
    finetuned_model_path="./results",
    prompts=None,
    max_length=100,
    num_samples=3,
    temperature=0.7,
    top_p=0.9,
):
    """
    Compare generation from the original and fine-tuned models.

    Args:
        original_model_name: Name of the original pre-trained model
        finetuned_model_path: Path to the fine-tuned model
        prompts: List of prompts to use (if None, default prompts will be used)
        max_length: Maximum length of generated text
        num_samples: Number of samples to generate for each prompt
        temperature: Temperature for sampling
        top_p: Top-p probability threshold for sampling
    """
    logger.info("Comparing original and fine-tuned models")

    # Default prompts if none are provided
    if prompts is None:
        prompts = [
            "To be or not to be,",
            "Friends, Romans, countrymen,",
            "Once upon a time",
            "The meaning of life is",
        ]

    # Load original model and tokenizer
    original_tokenizer = GPT2Tokenizer.from_pretrained(original_model_name)
    original_model = GPT2LMHeadModel.from_pretrained(original_model_name)

    # Load fine-tuned model and tokenizer
    finetuned_tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_path)
    finetuned_model = GPT2LMHeadModel.from_pretrained(finetuned_model_path)

    # Create generation pipelines
    original_generator = pipeline(
        "text-generation",
        model=original_model,
        tokenizer=original_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    finetuned_generator = pipeline(
        "text-generation",
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Generate and compare
    for i, prompt in enumerate(prompts):
        print(f'\nPrompt {i+1}: "{prompt}"')
        print("=" * 80)

        # Generate with original model
        print("\nOriginal GPT-2:")
        original_outputs = original_generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_samples,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        for j, output in enumerate(original_outputs):
            generated_text = output["generated_text"]
            print(f"\nSample {j+1}:")
            print(textwrap.fill(generated_text, width=80))

        # Generate with fine-tuned model
        print("\nFine-tuned GPT-2:")
        finetuned_outputs = finetuned_generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_samples,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        for j, output in enumerate(finetuned_outputs):
            generated_text = output["generated_text"]
            print(f"\nSample {j+1}:")
            print(textwrap.fill(generated_text, width=80))

        print("=" * 80)


def analyze_perplexity(
    model, tokenizer, texts, device=None, batch_size=4, max_length=512
):
    """
    Calculate perplexity of a model on given texts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        texts: List of text strings to evaluate on
        device: Device to use for computation
        batch_size: Batch size for processing
        max_length: Maximum sequence length

    Returns:
        Average perplexity across all texts
    """
    logger.info("Calculating perplexity")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # List to store perplexities
    perplexities = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize and truncate
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move to device
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        with torch.no_grad():
            # Get model outputs
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )

            # Calculate perplexity for each item in the batch
            for j in range(len(batch_texts)):
                # Get loss for this item
                loss = outputs.loss.item()

                # Calculate perplexity
                perplexity = torch.exp(torch.tensor(loss)).item()
                perplexities.append(perplexity)

                logger.debug(f"Text {i+j}: Perplexity = {perplexity:.2f}")

    # Calculate average perplexity
    avg_perplexity = np.mean(perplexities)
    logger.info(f"Average perplexity: {avg_perplexity:.2f}")

    return avg_perplexity


def compare_perplexity(
    original_model_name="gpt2",
    finetuned_model_path="./results",
    test_texts=None,
    domain_specific_texts=None,
):
    """
    Compare perplexity between original and fine-tuned models.

    Args:
        original_model_name: Name of the original model
        finetuned_model_path: Path to the fine-tuned model
        test_texts: General domain texts to test on
        domain_specific_texts: Domain-specific texts to test on
    """
    logger.info("Comparing perplexity between models")

    # Default test texts if none provided
    if test_texts is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning algorithms can be categorized as supervised or unsupervised.",
            "The stock market experienced significant volatility due to economic uncertainty.",
            "Scientists have discovered a new species of deep-sea fish that can glow in the dark.",
            "The annual music festival will feature artists from more than twenty countries.",
        ]

    # Default domain-specific texts if none provided
    if domain_specific_texts is None:
        # These are Shakespearean-style texts
        domain_specific_texts = [
            "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer.",
            "All the world's a stage, and all the men and women merely players.",
            "What light through yonder window breaks? It is the east, and Juliet is the sun.",
            "Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him.",
            "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
        ]

    # Load original model and tokenizer
    original_tokenizer = GPT2Tokenizer.from_pretrained(original_model_name)
    original_model = GPT2LMHeadModel.from_pretrained(original_model_name)

    # Load fine-tuned model and tokenizer
    finetuned_tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_path)
    finetuned_model = GPT2LMHeadModel.from_pretrained(finetuned_model_path)

    # Ensure padding token is set
    original_tokenizer.pad_token = original_tokenizer.eos_token
    finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token

    # Calculate perplexity on general domain texts
    print("\nPerplexity on general domain texts:")
    print("-" * 50)

    original_perplexity_general = analyze_perplexity(
        original_model, original_tokenizer, test_texts
    )

    finetuned_perplexity_general = analyze_perplexity(
        finetuned_model, finetuned_tokenizer, test_texts
    )

    # Calculate perplexity on domain-specific texts
    print("\nPerplexity on domain-specific texts:")
    print("-" * 50)

    original_perplexity_domain = analyze_perplexity(
        original_model, original_tokenizer, domain_specific_texts
    )

    finetuned_perplexity_domain = analyze_perplexity(
        finetuned_model, finetuned_tokenizer, domain_specific_texts
    )

    # Display results
    results = pd.DataFrame(
        {
            "Model": ["Original GPT-2", "Fine-tuned GPT-2"],
            "General Domain Perplexity": [
                f"{original_perplexity_general:.2f}",
                f"{finetuned_perplexity_general:.2f}",
            ],
            "Domain-Specific Perplexity": [
                f"{original_perplexity_domain:.2f}",
                f"{finetuned_perplexity_domain:.2f}",
            ],
        }
    )

    print("\nPerplexity Comparison:")
    print(results)

    # Create a bar chart to visualize the comparison
    plt.figure(figsize=(10, 6))

    x = np.arange(2)
    width = 0.35

    general_bars = plt.bar(
        x - width / 2,
        [original_perplexity_general, finetuned_perplexity_general],
        width,
        label="General Domain",
    )

    domain_bars = plt.bar(
        x + width / 2,
        [original_perplexity_domain, finetuned_perplexity_domain],
        width,
        label="Domain-Specific",
    )

    plt.xlabel("Model")
    plt.ylabel("Perplexity (lower is better)")
    plt.title("Perplexity Comparison: Original vs. Fine-tuned GPT-2")
    plt.xticks(x, ["Original GPT-2", "Fine-tuned GPT-2"])
    plt.legend()

    # Add values on top of bars
    for bar in general_bars + domain_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    # Print analysis
    print("\nAnalysis:")

    if finetuned_perplexity_domain < original_perplexity_domain:
        improvement = (
            1 - finetuned_perplexity_domain / original_perplexity_domain
        ) * 100
        print(
            f"- The fine-tuned model performs {improvement:.1f}% better on domain-specific texts."
        )
    else:
        print(
            "- The fine-tuned model does not show improvement on domain-specific texts."
        )

    if finetuned_perplexity_general > original_perplexity_general:
        degradation = (
            finetuned_perplexity_general / original_perplexity_general - 1
        ) * 100
        print(
            f"- The fine-tuned model performs {degradation:.1f}% worse on general domain texts."
        )
        print("  This suggests some degree of catastrophic forgetting or overfitting.")
    else:
        print(
            "- Interestingly, the fine-tuned model maintains good performance on general texts."
        )


def interactive_generation(model_path="./results", custom_prompts=None):
    """
    Interactive text generation with the fine-tuned model.

    Args:
        model_path: Path to the fine-tuned model
        custom_prompts: Optional list of custom prompts to suggest to the user
    """
    logger.info("Starting interactive generation")

    # Load fine-tuned model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Create a text generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # Default prompts if none provided
    if custom_prompts is None:
        custom_prompts = [
            "To be or not to be,",
            "Once upon a time",
            "The future of AI is",
            "In a galaxy far, far away",
        ]

    print("\nInteractive Text Generation with Fine-tuned GPT-2")
    print("=" * 60)
    print("Type a prompt and press Enter to generate text.")
    print("Type 'use X' to use suggested prompt X.")
    print("Type 'exit' to quit.")
    print("\nSuggested prompts:")

    for i, prompt in enumerate(custom_prompts):
        print(f"  {i+1}. {prompt}")

    # Interactive loop
    while True:
        try:
            # Get input
            user_input = input("\nEnter prompt: ")

            # Check for exit command
            if user_input.lower() == "exit":
                print("Exiting interactive mode.")
                break

            # Check for suggested prompt
            if user_input.lower().startswith("use "):
                try:
                    idx = int(user_input[4:]) - 1
                    if 0 <= idx < len(custom_prompts):
                        user_input = custom_prompts[idx]
                        print(f"Using prompt: {user_input}")
                    else:
                        print("Invalid prompt number")
                        continue
                except ValueError:
                    print("Invalid prompt number")
                    continue

            # Get generation parameters
            try:
                max_length = int(input("Max length (default=100): ") or "100")
                temperature = float(
                    input("Temperature (0.1-1.5, default=0.7): ") or "0.7"
                )
                top_p = float(input("Top-p (0.1-1.0, default=0.9): ") or "0.9")
                num_samples = int(input("Number of samples (default=1): ") or "1")
            except ValueError:
                print("Invalid input. Using default values.")
                max_length = 100
                temperature = 0.7
                top_p = 0.9
                num_samples = 1

            # Generate text
            print("\nGenerating...")
            start_time = time.time()

            outputs = generator(
                user_input,
                max_length=max_length,
                num_return_sequences=num_samples,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

            end_time = time.time()

            # Display results
            print(f"\nGeneration completed in {end_time - start_time:.2f} seconds.")
            for i, output in enumerate(outputs):
                if num_samples > 1:
                    print(f"\nSample {i+1}:")

                generated_text = output["generated_text"]
                print("-" * 60)
                print(textwrap.fill(generated_text, width=80))
                print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break

        except Exception as e:
            print(f"Error: {e}")


def main():
    """
    Main function to run the GPT-2 fine-tuning exercise.
    """
    print("Fine-tuning GPT-2")
    print("-" * 50)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Prepare the dataset
    print("\nStep 1: Preparing the dataset")
    dataset_path = prepare_dataset(
        dataset_name="tiny_shakespeare",
        use_subset=True,  # Use a subset for faster fine-tuning
        subset_size=1000,  # Adjust based on available compute
    )

    # Step 2: Fine-tune GPT-2
    print("\nStep 2: Fine-tuning GPT-2 on Shakespeare dataset")
    model_name = "gpt2"  # Can be "gpt2", "gpt2-medium", or "gpt2-large"
    output_dir = "./fine_tuned_gpt2_shakespeare"

    # Check if the model is already fine-tuned
    if os.path.exists(output_dir) and os.path.isfile(
        os.path.join(output_dir, "pytorch_model.bin")
    ):
        print(f"Found existing fine-tuned model at {output_dir}")
        print("Skipping fine-tuning. Set a different output_dir to fine-tune again.")
    else:
        model, tokenizer = finetune_gpt2(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_train_epochs=3,  # Use more epochs for better results
            per_device_train_batch_size=4,  # Adjust based on your GPU memory
            block_size=128,  # Context window size
        )

    # Step 3: Compare original vs fine-tuned model
    print("\nStep 3: Comparing original and fine-tuned models")
    # Shakespeare-specific prompts
    shakespeare_prompts = [
        "To be or not to be,",
        "What light through yonder window breaks?",
        "All the world's a stage,",
        "Friends, Romans, countrymen,",
    ]

    compare_models(
        original_model_name=model_name,
        finetuned_model_path=output_dir,
        prompts=shakespeare_prompts,
        max_length=100,
        num_samples=2,
    )

    # Step 4: Analyze perplexity
    print("\nStep 4: Analyzing perplexity")

    # Shakespeare-specific test texts
    shakespeare_texts = [
        "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer.",
        "All the world's a stage, and all the men and women merely players.",
        "What light through yonder window breaks? It is the east, and Juliet is the sun.",
        "Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him.",
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
    ]

    compare_perplexity(
        original_model_name=model_name,
        finetuned_model_path=output_dir,
        domain_specific_texts=shakespeare_texts,
    )

    # Step 5: Interactive generation
    print("\nStep 5: Interactive generation with the fine-tuned model")
    interactive_generation(model_path=output_dir, custom_prompts=shakespeare_prompts)

    print("\nExercise complete!")


if __name__ == "__main__":
    main()
