#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise 4: Building an AI Assistant

This exercise covers:
1. Connecting to a large language model API
2. Creating a conversational interface that maintains dialogue context
3. Implementing techniques for more effective prompting
4. Building a simple application that demonstrates practical use of LLMs

This exercise demonstrates how to build a simple AI assistant that can converse
with the user, maintain context over a conversation, and provide helpful responses.
It can be configured to use OpenAI's API or open-source alternatives.
"""

import os
import sys
import time
import json
import argparse
import textwrap
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure basic logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global variables
DEFAULT_MODEL = "gpt-3.5-turbo"  # Default OpenAI model
MAX_CONVERSATION_HISTORY = 10  # Maximum number of messages to keep in history


def setup_openai_api():
    """
    Set up the OpenAI API client. Checks for the API key in the environment.

    Returns:
        OpenAI client if setup was successful, None otherwise
    """
    try:
        import openai

        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try to get API key from user
            api_key = input("Please enter your OpenAI API key: ").strip()
            if not api_key:
                logger.error(
                    "No OpenAI API key provided. Set the OPENAI_API_KEY environment variable or provide it when prompted."
                )
                return None
            # Save to environment for this session
            os.environ["OPENAI_API_KEY"] = api_key

        # Set up the client
        openai.api_key = api_key
        logger.info("OpenAI API client set up successfully.")
        return openai

    except ImportError:
        logger.error(
            "OpenAI package not installed. Run 'pip install openai' to install it."
        )
        return None
    except Exception as e:
        logger.error(f"Error setting up OpenAI API: {e}")
        return None


def setup_huggingface_api():
    """
    Set up the Hugging Face API client. Checks for the API token in the environment.

    Returns:
        Inference API client if setup was successful, None otherwise
    """
    try:
        from huggingface_hub import InferenceClient

        # Check for API token
        api_token = os.environ.get("HF_API_TOKEN")
        if not api_token:
            # Try to get API token from user
            api_token = input(
                "Please enter your Hugging Face API token (or press Enter to use without token): "
            ).strip()
            # If still no token, we'll try to use the API without authentication
            if api_token:
                os.environ["HF_API_TOKEN"] = api_token

        # Set up the client
        client = InferenceClient(api_token)
        logger.info("Hugging Face API client set up successfully.")
        return client

    except ImportError:
        logger.error(
            "Hugging Face Hub package not installed. Run 'pip install huggingface_hub' to install it."
        )
        return None
    except Exception as e:
        logger.error(f"Error setting up Hugging Face API: {e}")
        return None


def query_openai_api(
    openai_client,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Tuple[str, float]:
    """
    Query the OpenAI API with a list of messages.

    Args:
        openai_client: The OpenAI client
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model to use for generation
        temperature: Controls randomness of the output
        max_tokens: Maximum number of tokens to generate

    Returns:
        Tuple of (generated text, response time in seconds)
    """
    try:
        start_time = time.time()

        # Call the API
        response = openai_client.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        end_time = time.time()
        response_time = end_time - start_time

        # Extract the generated text
        generated_text = response.choices[0].message["content"].strip()

        return generated_text, response_time

    except Exception as e:
        logger.error(f"Error querying OpenAI API: {e}")
        return f"Error: {str(e)}", 0.0


def query_huggingface_api(
    hf_client,
    messages: List[Dict[str, str]],
    model: str = "mistralai/Mistral-7B-Instruct-v0.1",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Tuple[str, float]:
    """
    Query the Hugging Face API with a list of messages.

    Args:
        hf_client: The Hugging Face Inference API client
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model to use for generation
        temperature: Controls randomness of the output
        max_tokens: Maximum number of tokens to generate

    Returns:
        Tuple of (generated text, response time in seconds)
    """
    try:
        start_time = time.time()

        # Convert messages to the format expected by the model
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                # System messages typically go at the beginning
                prompt += f"<system>\n{content}\n</system>\n"
            elif role == "user":
                prompt += f"<user>\n{content}\n</user>\n"
            elif role == "assistant":
                prompt += f"<assistant>\n{content}\n</assistant>\n"

        # Add the final assistant prompt
        prompt += "<assistant>\n"

        # Call the API
        response = hf_client.text_generation(
            prompt,
            model=model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            do_sample=True,
            stop_sequences=["</assistant>", "<user>"],
        )

        end_time = time.time()
        response_time = end_time - start_time

        # Extract and clean the generated text
        generated_text = response.strip()

        # Remove the assistant tag if it's included in the response
        if generated_text.endswith("</assistant>"):
            generated_text = generated_text[: -len("</assistant>")].strip()

        return generated_text, response_time

    except Exception as e:
        logger.error(f"Error querying Hugging Face API: {e}")
        return f"Error: {str(e)}", 0.0


class Conversation:
    """
    Class to manage a conversation with an LLM.
    """

    def __init__(
        self,
        api_type: str = "openai",
        model: str = DEFAULT_MODEL,
        system_message: Optional[str] = None,
        max_history: int = MAX_CONVERSATION_HISTORY,
    ):
        """
        Initialize a new conversation.

        Args:
            api_type: Type of API to use ("openai" or "huggingface")
            model: The model to use for generation
            system_message: Initial system message to set context
            max_history: Maximum number of messages to keep in history
        """
        self.api_type = api_type
        self.model = model
        self.max_history = max_history
        self.messages = []

        # Initialize the API client
        if api_type == "openai":
            self.client = setup_openai_api()
            if not self.client:
                raise ValueError("Failed to set up OpenAI API client.")
        elif api_type == "huggingface":
            self.client = setup_huggingface_api()
            if not self.client:
                raise ValueError("Failed to set up Hugging Face API client.")
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

        # Add system message if provided
        if system_message:
            self.add_system_message(system_message)

        # Initialize conversation metadata
        self.start_time = datetime.now()
        self.turn_count = 0

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            content: The content of the system message
        """
        system_message = {"role": "system", "content": content}

        # Check if there's an existing system message to replace
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = system_message
        else:
            self.messages.insert(0, system_message)

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The content of the user message
        """
        self.messages.append({"role": "user", "content": content})
        # Trim history if needed
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: The content of the assistant message
        """
        self.messages.append({"role": "assistant", "content": content})
        # Trim history if needed
        self._trim_history()

    def _trim_history(self) -> None:
        """
        Trim the conversation history to the maximum length.
        Always keeps the system message if present.
        """
        if len(self.messages) <= self.max_history:
            return

        # Preserve system message if present
        if self.messages[0]["role"] == "system":
            # Keep system message and (max_history - 1) most recent messages
            self.messages = [self.messages[0]] + self.messages[
                -(self.max_history - 1) :
            ]
        else:
            # Just keep the max_history most recent messages
            self.messages = self.messages[-self.max_history :]

    def get_response(
        self,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Tuple[str, float]:
        """
        Get a response from the LLM for the current conversation.

        Args:
            temperature: Controls randomness of the output
            max_tokens: Maximum number of tokens to generate

        Returns:
            Tuple of (assistant's response, response time in seconds)
        """
        self.turn_count += 1

        if self.api_type == "openai":
            response, response_time = query_openai_api(
                self.client,
                self.messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif self.api_type == "huggingface":
            response, response_time = query_huggingface_api(
                self.client,
                self.messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

        # Add the response to the conversation history
        self.add_assistant_message(response)

        return response, response_time

    def save_conversation(self, filename: Optional[str] = None) -> None:
        """
        Save the conversation to a JSON file.

        Args:
            filename: Name of the file to save to (default: conversation_<timestamp>.json)
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        # Prepare conversation data
        conversation_data = {
            "api_type": self.api_type,
            "model": self.model,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "turn_count": self.turn_count,
            "messages": self.messages,
        }

        # Save to file
        with open(filename, "w") as f:
            json.dump(conversation_data, f, indent=2)

        logger.info(f"Conversation saved to {filename}")

    def load_conversation(self, filename: str) -> None:
        """
        Load a conversation from a JSON file.

        Args:
            filename: Name of the file to load from
        """
        try:
            with open(filename, "r") as f:
                conversation_data = json.load(f)

            # Update conversation attributes
            self.api_type = conversation_data.get("api_type", self.api_type)
            self.model = conversation_data.get("model", self.model)
            self.messages = conversation_data.get("messages", [])
            self.turn_count = conversation_data.get("turn_count", 0)

            # Parse start_time if available
            start_time_str = conversation_data.get("start_time")
            if start_time_str:
                try:
                    self.start_time = datetime.fromisoformat(start_time_str)
                except ValueError:
                    self.start_time = datetime.now()

            logger.info(f"Conversation loaded from {filename}")

        except Exception as e:
            logger.error(f"Error loading conversation from {filename}: {e}")


def run_interactive_assistant(
    api_type: str = "openai",
    model: str = DEFAULT_MODEL,
    system_message: Optional[str] = None,
    max_history: int = MAX_CONVERSATION_HISTORY,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> None:
    """
    Run an interactive assistant in the console.

    Args:
        api_type: Type of API to use ("openai" or "huggingface")
        model: The model to use for generation
        system_message: Initial system message to set context
        max_history: Maximum number of messages to keep in history
        temperature: Controls randomness of the output
        max_tokens: Maximum number of tokens to generate
    """
    # Default system message if none provided
    if not system_message:
        system_message = """
        You are a helpful, honest, and concise AI assistant.
        Answer the user's questions or help them with tasks to the best of your ability.
        When you don't know something, admit it rather than making up information.
        Keep your responses relatively brief and to the point unless asked for more detail.
        """

    # Initialize conversation
    try:
        conversation = Conversation(
            api_type=api_type,
            model=model,
            system_message=system_message,
            max_history=max_history,
        )
    except ValueError as e:
        logger.error(f"Error initializing conversation: {e}")
        return

    # Print welcome message
    print("\n" + "=" * 80)
    print(f"Interactive AI Assistant using {api_type.upper()} API with model: {model}")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.")
    print("Type 'save' to save the conversation to a file.")
    print("Type 'system: <message>' to update the system message.")
    print("=" * 80 + "\n")

    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for special commands
            if user_input.lower() in ["exit", "quit"]:
                # Ask if the user wants to save the conversation
                save_response = (
                    input("Save conversation before exiting? (y/n): ").strip().lower()
                )
                if save_response in ["y", "yes"]:
                    filename = input(
                        "Enter filename (or press Enter for default): "
                    ).strip()
                    conversation.save_conversation(filename if filename else None)

                print("Goodbye!")
                break

            elif user_input.lower() == "save":
                filename = input(
                    "Enter filename (or press Enter for default): "
                ).strip()
                conversation.save_conversation(filename if filename else None)
                continue

            elif user_input.lower().startswith("system:"):
                # Update system message
                new_system_message = user_input[7:].strip()
                if new_system_message:
                    conversation.add_system_message(new_system_message)
                    print("System message updated.")
                else:
                    print("System message cannot be empty.")
                continue

            # Add user message to conversation
            conversation.add_user_message(user_input)

            # Get and display assistant's response
            print("\nAI Assistant is thinking...")
            response, response_time = conversation.get_response(
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Format and display the response
            print(f"\nAI Assistant ({response_time:.2f}s):")
            # Word wrap the response for better readability
            wrapped_response = textwrap.fill(response, width=80)
            print(wrapped_response)

        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected.")
            save_response = (
                input("Save conversation before exiting? (y/n): ").strip().lower()
            )
            if save_response in ["y", "yes"]:
                filename = input(
                    "Enter filename (or press Enter for default): "
                ).strip()
                conversation.save_conversation(filename if filename else None)

            print("Goodbye!")
            break

        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            print(f"\nAn error occurred: {e}")
            print("You can continue the conversation or type 'exit' to quit.")


def create_research_assistant() -> str:
    """
    Create a research assistant system message.

    Returns:
        System message for a research assistant
    """
    return """
    You are a research assistant with expertise in gathering, analyzing, and summarizing information.
    Your role is to help the user understand complex topics by providing accurate, well-structured information.
    
    When responding:
    1. Break down complex topics into understandable segments
    2. Cite sources where possible, clearly indicating what is factual and what is your analysis
    3. For technical subjects, provide clear explanations without unnecessary jargon
    4. If asked to summarize information, structure your response with bullet points or numbered lists
    5. When the user asks for multiple perspectives on a topic, present balanced viewpoints
    6. Always acknowledge limitations in current knowledge on a subject
    
    Avoid presenting speculation as fact. If you're uncertain about something, acknowledge the limits 
    of your knowledge.
    """


def create_coding_assistant() -> str:
    """
    Create a coding assistant system message.

    Returns:
        System message for a coding assistant
    """
    return """
    You are an expert coding assistant specializing in helping users write, debug, and understand code.
    Your expertise spans multiple programming languages and software development practices.
    
    When responding:
    1. Provide clean, well-commented code examples that follow best practices
    2. Explain your code line by line when the solution is complex
    3. Suggest efficient algorithms and data structures appropriate to the problem
    4. For debugging help, analyze the problem methodically and suggest specific fixes
    5. Recommend testing approaches where appropriate
    6. Add helpful comments within code explaining key concepts or tricky parts
    7. If there are multiple ways to solve a problem, briefly mention alternatives
    
    Always format code using proper code blocks with language specification.
    Focus on writing correct, secure, and maintainable code rather than quick hacks.
    """


def create_writing_assistant() -> str:
    """
    Create a writing assistant system message.

    Returns:
        System message for a writing assistant
    """
    return """
    You are a skilled writing assistant helping users improve their writing across various formats 
    and styles. Your expertise includes essays, articles, emails, creative writing, and more.
    
    When responding:
    1. Provide constructive feedback on clarity, structure, tone, and style
    2. Offer specific suggestions for improvement rather than vague advice
    3. Help maintain the user's authentic voice while enhancing their writing
    4. Adapt your guidance to match the intended audience and purpose of the text
    5. When asked to edit, explain your changes so the user can learn from them
    6. For creative writing, focus on strengthening narrative elements, character development, 
       and engaging language
    
    Respect the user's stylistic preferences while guiding them toward effective communication.
    Aim to be helpful without being overly critical or completely rewriting their work.
    """


def main():
    """
    Main function to run the AI assistant exercise.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run an interactive AI assistant.")

    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "huggingface"],
        default="openai",
        help="API to use: 'openai' or 'huggingface'",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (defaults depend on the API)",
    )

    parser.add_argument(
        "--assistant-type",
        type=str,
        choices=["general", "research", "coding", "writing"],
        default="general",
        help="Type of assistant: 'general', 'research', 'coding', or 'writing'",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0-1.0)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set default model based on API type
    if not args.model:
        if args.api == "openai":
            args.model = "gpt-3.5-turbo"
        else:  # huggingface
            args.model = "mistralai/Mistral-7B-Instruct-v0.1"

    # Get system message based on assistant type
    if args.assistant_type == "research":
        system_message = create_research_assistant()
    elif args.assistant_type == "coding":
        system_message = create_coding_assistant()
    elif args.assistant_type == "writing":
        system_message = create_writing_assistant()
    else:  # general
        system_message = None  # Will use the default in run_interactive_assistant

    # Run the interactive assistant
    run_interactive_assistant(
        api_type=args.api,
        model=args.model,
        system_message=system_message,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
