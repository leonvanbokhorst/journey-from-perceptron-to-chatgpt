#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify that the environment is set up correctly for Module 7.
This script checks if all required libraries are installed and can be imported.
"""

import sys
import importlib
import subprocess
import pkg_resources
from typing import List, Dict, Tuple

# List of required packages with minimum versions
REQUIRED_PACKAGES = {
    "torch": "1.10.0",
    "transformers": "4.18.0",
    "datasets": "2.0.0",
    "scikit-learn": "1.0.2",
    "matplotlib": "3.5.0",
    "pandas": "1.3.5",
    "nltk": "3.6.5",
    "huggingface_hub": "0.10.0",
    "tqdm": "4.62.3",
}

# OpenAI is optional as users might choose to use only Hugging Face
OPTIONAL_PACKAGES = {"openai": "0.27.0", "accelerate": "0.12.0"}


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and meets minimum version requirements.

    Args:
        package_name: Name of the package to check
        min_version: Minimum version required (optional)

    Returns:
        Tuple of (is_installed, message)
    """
    try:
        # Try to import the package
        module = importlib.import_module(package_name)

        # Get the installed version
        try:
            installed_version = pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            # Some packages use different import and pip names
            if package_name == "sklearn":
                installed_version = pkg_resources.get_distribution(
                    "scikit-learn"
                ).version
            else:
                installed_version = getattr(module, "__version__", "unknown")

        # Check if version meets requirements
        if min_version and pkg_resources.parse_version(
            installed_version
        ) < pkg_resources.parse_version(min_version):
            return (
                False,
                f"{package_name} version {installed_version} is installed but version {min_version} is required.",
            )

        return True, f"{package_name} version {installed_version} is installed."

    except ImportError:
        return False, f"{package_name} is not installed."


def check_cuda_availability() -> Tuple[bool, str]:
    """
    Check if CUDA is available through PyTorch.

    Returns:
        Tuple of (is_available, message)
    """
    try:
        import torch

        if torch.cuda.is_available():
            return (
                True,
                f"CUDA is available. Detected {torch.cuda.device_count()} CUDA device(s).",
            )
        else:
            return False, "CUDA is not available. GPU acceleration will not be used."
    except ImportError:
        return False, "PyTorch is not installed, cannot check CUDA availability."


def check_transformers_models() -> Tuple[bool, str]:
    """
    Check if Hugging Face Transformers can load model information.

    Returns:
        Tuple of (is_working, message)
    """
    try:
        from transformers import AutoConfig

        # Try to load a model config (lightweight operation)
        model_info = AutoConfig.from_pretrained(
            "bert-base-uncased", trust_remote_code=False
        )

        if model_info:
            return True, "Hugging Face Transformers is working correctly."
        else:
            return False, "Could not load model information from Hugging Face."

    except Exception as e:
        return False, f"Error testing Hugging Face Transformers: {str(e)}"


def main():
    """
    Main function to run all environment checks.
    """
    print("\n" + "=" * 80)
    print("Module 7 Environment Check".center(80))
    print("=" * 80 + "\n")

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")

    # Check required packages
    print("\nChecking required packages:")
    all_required_installed = True

    for package, min_version in REQUIRED_PACKAGES.items():
        is_installed, message = check_package(package, min_version)
        status = "✅" if is_installed else "❌"
        print(f"{status} {message}")

        if not is_installed:
            all_required_installed = False

    # Check optional packages
    print("\nChecking optional packages:")

    for package, min_version in OPTIONAL_PACKAGES.items():
        is_installed, message = check_package(package, min_version)
        status = "✅" if is_installed else "⚠️"
        print(f"{status} {message}")

    # Check CUDA
    print("\nChecking GPU support:")
    cuda_available, cuda_message = check_cuda_availability()
    cuda_status = "✅" if cuda_available else "⚠️"
    print(f"{cuda_status} {cuda_message}")

    # Check Transformers functionality
    print("\nChecking Hugging Face Transformers functionality:")
    transformers_working, transformers_message = check_transformers_models()
    transformers_status = "✅" if transformers_working else "❌"
    print(f"{transformers_status} {transformers_message}")

    # Summary
    print("\n" + "-" * 80)
    if all_required_installed and transformers_working:
        print("✅ All required dependencies are installed and working correctly!")
        print("You're ready to proceed with the Module 7 exercises.")
    else:
        print("❌ Some dependencies are missing or not working correctly.")
        print("Please install the missing packages using the following command:")
        print("\npip install -r requirements.txt\n")

    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
