#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 2: Text Generation with GPT-2

This exercise covers:
1. Using a pre-trained GPT-2 model for text generation
2. Experimenting with different prompts and generation parameters
3. Implementing few-shot learning through prompt engineering
4. Analyzing the quality and coherence of generated text

GPT-2 is a transformer-based language model pre-trained on a diverse corpus
of internet text. It excels at generating coherent and contextually relevant text
continuations based on given prompts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    pipeline,
    set_seed,
)
import time
import re
from nltk.tokenize import sent_tokenize
import pandas as pd
from IPython.display import display

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
set_seed(42)

# Try to import and download NLTK data if not already present
try:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt", quiet=True)
except ImportError:
    print("NLTK not installed. Using simple sentence splitting instead.")
    sent_tokenize = lambda text: re.findall(r"[^.!?]+[.!?]", text)


def load_gpt2_model(model_name="gpt2", device=None):
    """
    Load a pre-trained GPT-2 model and tokenizer.

    Args:
        model_name: Which GPT-2 model to load ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        device: Device to use (None for auto-detection)

    Returns:
        model: The loaded GPT-2 model
        tokenizer: The GPT-2 tokenizer
    """
    print(f"Loading {model_name} model and tokenizer...")

    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Move model to the appropriate device
    model.to(device)

    # Set the padding token to be the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    num_return=1,
    device=None,
):
    """
    Generate text using the GPT-2 model.

    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        prompt: Input text to continue from
        max_length: Maximum length of the generated text
        temperature: Controls randomness (lower = more focused, higher = more diverse)
        top_k: Keep only top k tokens with highest probability
        top_p: Keep tokens with cumulative probability >= top_p
        num_return: Number of text sequences to generate
        device: Device to use for generation

    Returns:
        List of generated text sequences
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Move input to device
    if device:
        input_ids = input_ids.to(device)

    # Configure generation parameters
    gen_config = {
        "input_ids": input_ids,
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_return_sequences": num_return,
        "do_sample": True,
        "no_repeat_ngram_size": 2,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Generate text
    output_sequences = model.generate(**gen_config)

    # Decode and return the output
    generated_texts = [
        tokenizer.decode(sequence, skip_special_tokens=True)
        for sequence in output_sequences
    ]

    return generated_texts


def basic_generation_example(model, tokenizer, device=None):
    """
    Demonstrate basic text generation with GPT-2.

    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        device: Device to use for generation
    """
    print("\nBasic Text Generation Example")
    print("-" * 50)

    # Define prompts
    prompts = [
        "Once upon a time in a land far away",
        "The future of artificial intelligence is",
        "Scientists have discovered a new species that",
        "The most important thing to remember about programming is",
    ]

    # Generate text for each prompt
    for i, prompt in enumerate(prompts):
        print(f'\nPrompt {i+1}: "{prompt}"')
        generated_texts = generate_text(
            model, tokenizer, prompt, max_length=100, device=device
        )

        print("\nGenerated Text:")
        print(generated_texts[0])
        print("-" * 50)


def experiment_with_parameters(model, tokenizer, device=None):
    """
    Experiment with different generation parameters.

    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        device: Device to use for generation
    """
    print("\nExperimenting with Generation Parameters")
    print("-" * 50)

    # Sample prompt
    prompt = "The solution to climate change requires"

    print(f'Prompt: "{prompt}"\n')

    # Test different temperatures
    print("Varying temperature (controls randomness):")
    temperatures = [0.3, 0.7, 1.0, 1.5]

    results = []
    for temp in temperatures:
        generated_text = generate_text(
            model, tokenizer, prompt, temperature=temp, max_length=75, device=device
        )[0]

        # Truncate to a reasonable length for display
        display_text = generated_text[len(prompt) : 100 + len(prompt)]

        results.append(
            {
                "Temperature": temp,
                "Generated Text": display_text,
            }
        )

    # Display as a DataFrame
    df = pd.DataFrame(results)
    display(df)

    # Test different top_k values
    print("\nVarying top_k (controls diversity of word choices):")
    top_ks = [5, 20, 50, 100]

    results = []
    for k in top_ks:
        generated_text = generate_text(
            model, tokenizer, prompt, top_k=k, max_length=75, device=device
        )[0]

        # Truncate to a reasonable length for display
        display_text = generated_text[len(prompt) : 100 + len(prompt)]

        results.append(
            {
                "Top K": k,
                "Generated Text": display_text,
            }
        )

    # Display as a DataFrame
    df = pd.DataFrame(results)
    display(df)

    # Test different top_p values
    print("\nVarying top_p (nucleus sampling parameter):")
    top_ps = [0.5, 0.8, 0.92, 0.99]

    results = []
    for p in top_ps:
        generated_text = generate_text(
            model, tokenizer, prompt, top_p=p, max_length=75, device=device
        )[0]

        # Truncate to a reasonable length for display
        display_text = generated_text[len(prompt) : 100 + len(prompt)]

        results.append(
            {
                "Top P": p,
                "Generated Text": display_text,
            }
        )

    # Display as a DataFrame
    df = pd.DataFrame(results)
    display(df)


def implement_few_shot_learning(model, tokenizer, device=None):
    """
    Demonstrate few-shot learning through prompt engineering.

    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        device: Device to use for generation
    """
    print("\nFew-Shot Learning with GPT-2")
    print("-" * 50)

    # Example 1: Sentiment classification
    print("Example 1: Sentiment Classification")
    sentiment_prompt = """
Review: This movie was fantastic! The acting was superb and I was engaged throughout.
Sentiment: Positive

Review: I was extremely disappointed by this restaurant. The service was slow and the food was bland.
Sentiment: Negative

Review: The laptop works fine for basic tasks, but it's nothing special.
Sentiment: Neutral

Review: The concert was incredible, one of the best I've ever attended!
Sentiment: 
"""

    print("\nPrompt:")
    print(sentiment_prompt)

    sentiment_response = generate_text(
        model,
        tokenizer,
        sentiment_prompt,
        max_length=len(tokenizer.encode(sentiment_prompt)) + 5,
        temperature=0.3,
        device=device,
    )[0]

    # Extract only the generated part
    generated_part = sentiment_response[len(sentiment_prompt) :]
    print("\nGPT-2 Response (generated part only):")
    print(generated_part.strip())

    # Example 2: Entity extraction
    print("\nExample 2: Entity Extraction")
    entity_prompt = """
Text: Apple Inc. is planning to open a new headquarters in Austin, Texas by the end of 2022.
Entities: Apple Inc. (ORGANIZATION), Austin (LOCATION), Texas (LOCATION), 2022 (DATE)

Text: President Joe Biden met with UK Prime Minister Rishi Sunak to discuss climate change policy.
Entities: Joe Biden (PERSON), UK (LOCATION), Rishi Sunak (PERSON), climate change (TOPIC)

Text: The Mars rover Perseverance sent back stunning images of the Martian surface yesterday.
Entities: 
"""

    print("\nPrompt:")
    print(entity_prompt)

    entity_response = generate_text(
        model,
        tokenizer,
        entity_prompt,
        max_length=len(tokenizer.encode(entity_prompt)) + 40,
        temperature=0.3,
        device=device,
    )[0]

    # Extract only the generated part
    generated_part = entity_response[len(entity_prompt) :]
    print("\nGPT-2 Response (generated part only):")
    print(generated_part.strip())

    # Example 3: Format conversion
    print("\nExample 3: Format Conversion (JSON)")
    format_prompt = """
Convert the following text to JSON format:

Text: John Smith, a 42-year-old software engineer from San Francisco, has been hired by Google.
JSON: {"name": "John Smith", "age": 42, "profession": "software engineer", "location": "San Francisco", "event": "hired", "company": "Google"}

Text: The new Toyota Prius, priced at $30,000, gets 55 miles per gallon and comes in three colors.
JSON: {"product": "Toyota Prius", "price": "$30,000", "efficiency": "55 miles per gallon", "color_options": 3}

Text: The restaurant is open Tuesday through Sunday from 5PM to 10PM, with a special brunch on weekends from 10AM to 2PM.
JSON: 
"""

    print("\nPrompt:")
    print(format_prompt)

    format_response = generate_text(
        model,
        tokenizer,
        format_prompt,
        max_length=len(tokenizer.encode(format_prompt)) + 100,
        temperature=0.2,
        device=device,
    )[0]

    # Extract only the generated part
    generated_part = format_response[len(format_prompt) :]
    print("\nGPT-2 Response (generated part only):")
    print(generated_part.strip())


def analyze_generated_text(model, tokenizer, device=None):
    """
    Analyze the quality and coherence of generated text.

    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        device: Device to use for generation
    """
    print("\nAnalyzing Generated Text")
    print("-" * 50)

    # Generate a longer piece of text
    prompt = "The history of artificial intelligence can be traced back to"

    print(f'Prompt: "{prompt}"\n')

    # Generate a long text with low temperature for coherence
    generated_text = generate_text(
        model, tokenizer, prompt, max_length=500, temperature=0.7, device=device
    )[0]

    # Print the generated text
    print("Generated text:")
    print(generated_text)

    # Basic analysis: Split into sentences
    sentences = sent_tokenize(generated_text)

    print(f"\nNumber of sentences: {len(sentences)}")
    print(
        f"Average sentence length: {np.mean([len(s.split()) for s in sentences]):.1f} words"
    )

    # Calculate token probabilities for the generated text
    input_ids = tokenizer.encode(generated_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get probabilities for the next token after each position
    probs = torch.softmax(logits, dim=-1)

    # For each token (except the last), get the probability of the actual next token
    token_probs = []
    for i in range(input_ids.shape[1] - 1):
        next_token_id = input_ids[0, i + 1].item()
        next_token_prob = probs[0, i, next_token_id].item()
        token_probs.append(next_token_prob)

    # Calculate statistics about token probabilities
    avg_prob = np.mean(token_probs)
    min_prob = np.min(token_probs)

    print(f"\nAverage next-token probability: {avg_prob:.4f}")
    print(f"Minimum next-token probability: {min_prob:.4f}")

    # Find the 5 tokens with lowest probability
    token_texts = [
        tokenizer.decode([input_ids[0, i + 1].item()]) for i in range(len(token_probs))
    ]
    lowest_probs_idx = np.argsort(token_probs)[:5]

    print("\nTokens with lowest probability:")
    for idx in lowest_probs_idx:
        print(f"  '{token_texts[idx]}' - Probability: {token_probs[idx]:.4f}")

    # Visualization of probabilities
    plt.figure(figsize=(10, 6))
    plt.plot(token_probs)
    plt.xlabel("Token Position")
    plt.ylabel("Probability")
    plt.title("Next-Token Probabilities Throughout Generated Text")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Subjective analysis
    print("\nSubjective Analysis:")
    print("1. Coherence: Does the text stay on topic and make logical sense?")
    print("2. Factual accuracy: Are the facts presented in the text accurate?")
    print("3. Writing style: How natural is the writing style?")
    print("4. Transitions: Are there abrupt shifts in topic or tone?")
    print("\nTake a moment to assess these aspects in the generated text.")


def creative_writing_prompter(model, tokenizer, device=None):
    """
    Interactive prompter for creative writing tasks.

    Args:
        model: The GPT-2 model
        tokenizer: The GPT-2 tokenizer
        device: Device to use for generation
    """
    print("\nCreative Writing with GPT-2")
    print("-" * 50)
    print("This tool helps you generate creative text with GPT-2.")
    print("You can provide a prompt, and customize the generation parameters.\n")

    # Sample prompts
    sample_prompts = [
        "Write a short story about a robot who discovers emotions",
        "Create a poem about the changing seasons",
        "Describe an alien world with unusual physics",
        "Write a dialogue between a time traveler and a historical figure",
        "Custom prompt (enter your own)",
    ]

    # Display sample prompts
    print("Sample prompts:")
    for i, prompt in enumerate(sample_prompts):
        print(f"  {i+1}. {prompt}")

    # Get user choice
    try:
        choice = int(input("\nEnter your choice (1-5): "))
        if choice < 1 or choice > 5:
            raise ValueError("Invalid choice")
    except ValueError:
        print("Invalid selection, defaulting to option 1.")
        choice = 1

    # Process choice
    if choice == 5:
        prompt = input("\nEnter your custom prompt: ")
    else:
        prompt = sample_prompts[choice - 1]

    # Get generation parameters
    print("\nGeneration parameters (press Enter for defaults):")

    try:
        length_input = input("Max generation length (50-1000, default=200): ")
        max_length = int(length_input) if length_input else 200
        max_length = max(50, min(1000, max_length))

        temp_input = input("Temperature (0.1-1.5, default=0.8): ")
        temperature = float(temp_input) if temp_input else 0.8
        temperature = max(0.1, min(1.5, temperature))

        top_p_input = input("Top-p (0.1-1.0, default=0.9): ")
        top_p = float(top_p_input) if top_p_input else 0.9
        top_p = max(0.1, min(1.0, top_p))
    except ValueError:
        print("Invalid input, using default values.")
        max_length = 200
        temperature = 0.8
        top_p = 0.9

    print(f'\nGenerating text for prompt: "{prompt}"')
    print(
        f"Parameters: max_length={max_length}, temperature={temperature:.1f}, top_p={top_p:.1f}"
    )

    # Generate text
    start_time = time.time()
    generated_texts = generate_text(
        model,
        tokenizer,
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return=1,
        device=device,
    )
    end_time = time.time()

    # Display result
    print(f"\nGeneration completed in {end_time - start_time:.2f} seconds.")
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_texts[0])
    print("-" * 50)

    print(
        "\nTip: You can copy this text and use it as inspiration or a starting point for your writing."
    )


def main():
    """
    Main function to run the GPT-2 text generation exercise.
    """
    print("Text Generation with GPT-2")
    print("-" * 50)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    # Note: You can use "gpt2-medium" or "gpt2-large" for better results,
    # but they require more GPU memory and are slower on CPU
    model, tokenizer, device = load_gpt2_model("gpt2", device)

    # Run examples
    basic_generation_example(model, tokenizer, device)
    experiment_with_parameters(model, tokenizer, device)
    implement_few_shot_learning(model, tokenizer, device)
    analyze_generated_text(model, tokenizer, device)
    creative_writing_prompter(model, tokenizer, device)

    print("\nExercise complete!")


if __name__ == "__main__":
    main()
