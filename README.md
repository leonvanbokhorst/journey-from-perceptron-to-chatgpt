# 🚀 Journey from Perceptron to ChatGPT

Welcome to an exciting hands-on exploration of neural networks! This repository traces the evolution of artificial intelligence from the humble perceptron of the 1950s to today's powerful large language models like ChatGPT.

Whether you're a beginner curious about AI fundamentals or an experienced practitioner wanting to strengthen your understanding of neural network foundations, this curriculum offers a structured learning path with practical coding exercises at every step.

## 🧠 Learning Journey

This curriculum is designed as a progressive journey through the key innovations that transformed neural networks from simple binary classifiers to sophisticated language models:

1. [The Perceptron and Early Neural Networks](exercises/module1/README.md)
2. [Multi-Layer Perceptrons and Backpropagation](exercises/module2/README.md)
3. [Recurrent Neural Networks and Sequence Modeling](exercises/module3/README.md)
4. [Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)](exercises/module4/README.md)
5. [Sequence-to-Sequence Learning and Attention Mechanisms](exercises/module5/README.md)
6. [The Transformer Architecture – "Attention Is All You Need"](exercises/module6/README.md)
7. [Pre-trained Transformers and ChatGPT](exercises/module7/README.md)

Each module builds upon the previous ones, gradually introducing more sophisticated concepts while reinforcing your understanding through hands-on implementation.

## 🔧 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of Python and machine learning concepts

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/leonvanbokhorst/journey-from-perceptron-to-chatgpt.git
   cd journey-from-perceptron-to-chatgpt
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Exercises

Each exercise is contained in a Python file that can be run directly:

```bash
python exercises/module1/code/ex_01_perceptron_basics.py
```

For visualization exercises, the scripts will generate plots that show up in a window or are saved to disk.

## 📂 Repository Structure

Each module follows a consistent structure:

- **code/**: Python implementation files for each exercise
- **guides/**: Markdown files explaining the concepts and theory
- **data/**: Data files used in the exercises (where applicable)
- **README.md**: Overview, learning objectives, and instructions for the module

## 📚 Module Highlights

### [Module 1: The Perceptron and Early Neural Networks](exercises/module1/README.md)

- Build a perceptron algorithm from scratch with visualization
- Implement multi-class classification strategies
- Visualize decision boundaries and understand perceptron limitations
- Create your first neural network using PyTorch

### [Module 2: Multi-Layer Perceptrons and Backpropagation](exercises/module2/README.md)

- Implement MLPs and backpropagation from scratch
- Explore different activation functions and their effects
- Build classification models with configurable architectures
- Leverage PyTorch for efficient neural network training

### [Module 3: Recurrent Neural Networks and Sequence Modeling](exercises/module3/README.md)

- Create RNN cells from scratch to process sequential data
- Apply RNNs to time series prediction challenges
- Build a character-level language model for text generation
- Implement efficient RNNs using PyTorch's built-in modules

### [Module 4: Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)](exercises/module4/README.md)

- Implement LSTM and GRU architectures from scratch
- Visualize and understand gating mechanisms
- Compare performance of different RNN variants
- Create bidirectional networks for sentiment analysis

### [Module 5: Sequence-to-Sequence Learning and Attention Mechanisms](exercises/module5/README.md)

- Build encoder-decoder architectures for sequence transformation
- Implement and visualize attention mechanisms
- Create neural machine translation systems
- Explore advanced attention variants (foundation for Transformers)

### [Module 6: The Transformer Architecture – "Attention Is All You Need"](exercises/module6/README.md)

- Implement self-attention and multi-head attention mechanisms
- Build Transformer encoder and decoder blocks
- Create a complete Transformer for sequence-to-sequence tasks
- Analyze attention patterns through visualization

### [Module 7: Pre-trained Transformers and ChatGPT](exercises/module7/README.md)

- Fine-tune BERT models for text classification
- Generate text with pre-trained GPT-2
- Customize language models for specific domains
- Build a conversational AI assistant with modern LLM APIs

## 🤝 Contributing

Contributions are welcome! If you find bugs, have suggestions for improvements, or want to add enhancements:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This curriculum draws inspiration from the historical development of neural network architectures
- Special thanks to the research pioneers whose papers are referenced throughout the modules
- Built with love for the AI community and lifelong learners everywhere
