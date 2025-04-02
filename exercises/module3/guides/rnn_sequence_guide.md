# Recurrent Neural Networks and Sequence Modeling

## Introduction

While feedforward neural networks excel at many tasks, they have a significant limitation: they process each input independently, without any memory of previous inputs. This makes them unsuitable for sequential data like text, speech, time series, or any data where the order and context matter.

Recurrent Neural Networks (RNNs) were designed to address this limitation by introducing the concept of "memory" into neural networks. They maintain an internal state (or "memory") that captures information about what has been processed so far, enabling them to learn from sequential data.

## The Need for Sequence Modeling

Many real-world data are inherently sequential:

- **Natural Language**: Words in a sentence follow a specific order; the meaning of a word often depends on previous words.
- **Time Series**: Stock prices, weather data, and sensor readings evolve over time, with patterns and dependencies.
- **Speech**: Audio signals are sequences of sound waves where temporal patterns matter.
- **Genomics**: DNA sequences are ordered chains of nucleotides.

Traditional feedforward networks treat each input as independent, losing the sequential information. But in these domains, the order and context are crucial to understanding the data.

## RNN Architecture

### Basic Recurrent Structure

At its core, an RNN processes sequences by:

1. Taking the current input (x<sub>t</sub>) **and** the previous hidden state (h<sub>t-1</sub>)
2. Computing a new hidden state (h<sub>t</sub>)
3. Optionally producing an output (y<sub>t</sub>)
4. Passing the hidden state to the next time step

This recurrent connection allows information to persist, creating a form of "memory" in the network.

### Mathematical Formulation

The basic RNN update can be expressed as:

h<sub>t</sub> = tanh(W<sub>h</sub>h<sub>t-1</sub> + W<sub>x</sub>x<sub>t</sub> + b<sub>h</sub>)

y<sub>t</sub> = W<sub>y</sub>h<sub>t</sub> + b<sub>y</sub>

Where:

- h<sub>t</sub> is the hidden state at time t
- x<sub>t</sub> is the input at time t
- W<sub>h</sub>, W<sub>x</sub>, W<sub>y</sub> are weight matrices
- b<sub>h</sub>, b<sub>y</sub> are bias terms

### Unfolding in Time

An RNN can be "unfolded" across time, creating a computational graph that resembles a very deep feedforward network with shared weights. This perspective helps understand both the power and the challenges of training RNNs.

## Types of RNN Architectures

### Many-to-Many

- **Sequence-to-Sequence**: Inputs and outputs at each time step
- **Applications**: Part-of-speech tagging, real-time translation

### Many-to-One

- **Processes a sequence but produces a single output**
- **Applications**: Sentiment analysis, document classification

### One-to-Many

- **Takes a single input and generates a sequence**
- **Applications**: Image captioning, music generation

### Bidirectional RNNs

- **Process sequences in both forward and backward directions**
- **Advantage**: Can capture context from both past and future time steps
- **Applications**: Speech recognition, machine translation

## Training RNNs

### Backpropagation Through Time (BPTT)

RNNs are trained using an extension of backpropagation called Backpropagation Through Time (BPTT). This algorithm:

1. Unfolds the RNN for a specific number of time steps
2. Computes the loss (e.g., cross-entropy for classification)
3. Propagates gradients backward through time
4. Updates weights using these gradients

### Challenges in Training RNNs

#### Vanishing Gradients

- As gradients are propagated backward through many time steps, they can become extremely small.
- This makes it difficult for RNNs to learn long-range dependencies.
- Early parts of a sequence have diminishing influence on later predictions.

#### Exploding Gradients

- Conversely, gradients can also grow exponentially during backpropagation.
- This can lead to numerical instability and very large weight updates.
- Common solution: Gradient clipping (limiting the magnitude of gradients).

## Character-Level Language Models

One classic application of RNNs is character-level language modeling:

1. The network receives one character at a time from a text corpus.
2. It's trained to predict the next character in the sequence.
3. The hidden state captures patterns and dependencies in language.
4. Once trained, sampling from the model can generate new text character by character.

This simple model demonstrates the power of RNNs to capture sequential patterns and dependencies in natural language text.

## Limitations of Simple RNNs

While basic RNNs are conceptually powerful, they suffer from:

1. **Limited Memory Span**: Difficulty capturing dependencies over long sequences
2. **Vanishing/Exploding Gradients**: Numerical issues that harm training
3. **Computational Inefficiency**: Sequential processing limits parallelization

These limitations led to the development of more advanced architectures like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units), which we'll explore in the next module.

## Practical Considerations

### Input Representation

Sequential data needs appropriate representation:

- **Text**: One-hot encoding, embeddings
- **Time Series**: Normalization, windowing
- **Categorical Sequences**: Encoding strategies

### Sequence Length

- **Fixed-Length**: Padding shorter sequences, truncating longer ones
- **Variable-Length**: More complex but often more appropriate

### Batch Processing

- Mini-batch training with sequences of similar length
- Padding and masking to handle variable-length sequences

## Next Steps

In the following exercises, we'll implement RNNs from scratch and explore their capabilities for sequence modeling tasks. In the next module, we'll address the limitations of simple RNNs by exploring more sophisticated architectures like LSTMs and GRUs.
