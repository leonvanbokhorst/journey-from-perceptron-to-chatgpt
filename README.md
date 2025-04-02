# Journey from Perceptron to ChatGPT

This repository contains hands-on exercises for the curriculum that traces the evolution of neural networks from the simple perceptron to modern transformer-based models like ChatGPT.

## Overview

The exercises are organized by module, following the curriculum structure:

1. The Perceptron and Early Neural Networks ✅
2. Multi-Layer Perceptrons and Backpropagation ✅
3. Recurrent Neural Networks and Sequence Modeling ✅
4. Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) ✅
5. Sequence-to-Sequence Learning and Attention Mechanisms ✅
6. The Transformer Architecture – "Attention Is All You Need" ✅
7. Pre-trained Transformers and ChatGPT ✅

Each module contains Python code files, concept guides, and supplementary data to reinforce learning.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of Python and machine learning concepts

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/journey-from-perceptron-to-chatgpt.git
   cd journey-from-perceptron-to-chatgpt
   ```

2. Create a virtual environment (recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```
python exercises/module1/code/ex_01_perceptron_basics.py
```

For visualization exercises, the scripts will generate plots that show up in a window or are saved to disk.

## Structure

Each module directory contains:

- **code/**: Python implementation files for each exercise
- **guides/**: Markdown files explaining the concepts and theory
- **data/**: Data files used in the exercises (where applicable)

## Completed Modules

### Module 1: The Perceptron and Early Neural Networks

- Basic perceptron implementation with visualization
- Multi-class classification with perceptrons
- Decision boundary visualization
- PyTorch implementation of perceptrons

### Module 2: Multi-Layer Perceptrons and Backpropagation

- MLP implementation with backpropagation from scratch
- Exploration of different activation functions
- Classification with MLPs (comparison of architectures and mini-batch training)
- PyTorch implementation of MLPs with automatic differentiation

### Module 3: Recurrent Neural Networks and Sequence Modeling

- Basic RNN implementation from scratch with NumPy
- Time series prediction using RNNs
- Character-level language model for text generation
- RNN implementations using PyTorch, including a manual RNN cell and LSTM

### Module 4: Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)

- LSTM implementation from scratch with NumPy
- GRU implementation from scratch with detailed visualization of gates
- Comparative performance analysis of RNNs, LSTMs, and GRUs on sequence prediction tasks
- Sentiment analysis with bidirectional LSTM/GRU using PyTorch

### Module 5: Sequence-to-Sequence Learning and Attention Mechanisms

- Basic sequence-to-sequence model implementation from scratch
- Attention mechanism integration with sequence-to-sequence models
- Implementation of key Transformer components (multi-head attention, positional encoding)
- Neural machine translation with attention visualization

### Module 6: The Transformer Architecture – "Attention Is All You Need"

- Implementation of scaled dot-product self-attention mechanisms with visualization
- Multi-head attention implementation with analysis of how different heads learn
- Full Transformer encoder stack with representation visualization
- Complete encoder-decoder Transformer for sequence-to-sequence tasks

### Module 7: Pre-trained Transformers and ChatGPT

- Fine-tuning BERT for text classification and sentiment analysis
- Text generation with pre-trained GPT-2 models and parameter experimentation
- Fine-tuning GPT-2 on specialized corpora with performance comparison
- Building a complete AI assistant application with LLM API integration

## Contributing

Feel free to submit pull requests or open issues if you find any problems or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
