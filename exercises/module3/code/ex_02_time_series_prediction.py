#!/usr/bin/env python3
"""
Time Series Prediction with RNNs - Exercise 2
Apply RNNs to time series forecasting tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import urllib.request
import zipfile
import io


# Simplified RNN class for time series prediction
class RNN:
    """
    A simple Recurrent Neural Network for time series forecasting.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the RNN with the given sizes."""
        # Xavier/Glorot initialization for weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01

        # Initialize biases as zeros
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Store dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Reset internal states
        self.reset_states()

    def reset_states(self):
        """Reset the states stored for backpropagation through time."""
        self.last_inputs = {}
        self.last_hs = {}
        self.last_outputs = {}

        # Initial hidden state is set to zero
        self.h = np.zeros((self.hidden_size, 1))

    def forward(self, x, h_prev=None):
        """Forward pass for a single time step."""
        if h_prev is None:
            h_prev = self.h

        # Compute hidden state
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)

        # Compute output
        y = np.dot(self.Why, h) + self.by

        # Update hidden state
        self.h = h

        return y, h

    def forward_sequence(self, x_sequence):
        """Forward pass for a sequence of inputs."""
        # Reset states at the beginning of each sequence
        self.reset_states()

        outputs = []
        hidden_states = []

        for t, x in enumerate(x_sequence):
            # Reshape x to be a column vector
            x = x.reshape(-1, 1)

            # Store for backpropagation
            self.last_inputs[t] = x
            self.last_hs[t] = self.h.copy()

            # Forward pass for single time step
            y, h = self.forward(x)

            # Store output and hidden state
            self.last_outputs[t] = y
            outputs.append(y)
            hidden_states.append(h)

        return outputs, hidden_states

    def backward(self, targets, learning_rate=0.01, clip_value=5.0):
        """Backpropagation through time (BPTT) algorithm."""
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # Initialize loss
        loss = 0
        dh_next = np.zeros_like(self.h)

        # Backpropagate through time
        for t in reversed(range(len(targets))):
            # Current target
            target = targets[t].reshape(-1, 1)

            # Current output
            y = self.last_outputs[t]

            # Compute loss (mean squared error)
            loss += np.sum((y - target) ** 2) / 2

            # Gradient of output
            dy = y - target

            # Gradient of Why
            dWhy += np.dot(dy, self.last_hs[t].T)
            dby += dy

            # Gradient of hidden state
            dh = np.dot(self.Why.T, dy) + dh_next

            # Gradient of tanh
            dh_raw = (1 - self.last_hs[t] ** 2) * dh

            # Gradient of biases
            dbh += dh_raw

            # Gradient of Whh
            dWhh += (
                np.dot(dh_raw, self.last_hs[t - 1].T)
                if t > 0
                else np.dot(dh_raw, np.zeros((self.hidden_size, 1)).T)
            )

            # Gradient of Wxh
            dWxh += np.dot(dh_raw, self.last_inputs[t].T)

            # Gradient of next hidden state
            dh_next = np.dot(self.Whh.T, dh_raw)

        # Average loss
        loss /= len(targets)

        # Clip gradients to prevent explosion
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Update weights
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        return loss

    def predict(self, x_sequence):
        """Make predictions for a sequence of inputs."""
        outputs, _ = self.forward_sequence(x_sequence)
        return outputs


def fetch_data():
    """
    Fetch and prepare time series data.

    This function downloads a dataset if not already available locally.
    It provides options for different time series datasets.

    Returns:
        DataFrame with time series data
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join("exercises", "module3", "data")
    os.makedirs(data_dir, exist_ok=True)

    # We'll use a simple synthetic dataset first, then some real data

    # Option 1: Synthetic data
    # =======================
    np.random.seed(42)

    # Creating a synthetic time series with patterns
    t = np.linspace(0, 4 * np.pi, 1000)

    # Trend + Seasonal + Noise
    trend = 0.05 * t
    seasonal_1 = 2 * np.sin(t)
    seasonal_2 = 0.5 * np.sin(5 * t)
    noise = 0.5 * np.random.randn(len(t))

    ts_synthetic = trend + seasonal_1 + seasonal_2 + noise

    df_synthetic = pd.DataFrame(
        {
            "date": pd.date_range(
                start="2020-01-01", periods=len(ts_synthetic), freq="D"
            ),
            "value": ts_synthetic,
        }
    )
    df_synthetic.set_index("date", inplace=True)

    # Option 2: Air Passengers dataset
    # ==============================
    air_passengers_file = os.path.join(data_dir, "AirPassengers.csv")

    if not os.path.exists(air_passengers_file):
        print("Downloading Air Passengers dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        urllib.request.urlretrieve(url, air_passengers_file)

    df_air = pd.read_csv(air_passengers_file)
    df_air["Month"] = pd.to_datetime(df_air["Month"])
    df_air.set_index("Month", inplace=True)
    df_air.columns = ["value"]  # Rename for consistency

    # Option 3: Daily temperature dataset
    # ================================
    temp_file = os.path.join(data_dir, "daily-min-temperatures.csv")

    if not os.path.exists(temp_file):
        print("Downloading daily temperature dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
        urllib.request.urlretrieve(url, temp_file)

    df_temp = pd.read_csv(temp_file)
    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
    df_temp.set_index("Date", inplace=True)
    df_temp.columns = ["value"]  # Rename for consistency

    return {"synthetic": df_synthetic, "air_passengers": df_air, "temperature": df_temp}


def prepare_time_series_data(
    data, sequence_length=10, prediction_steps=1, train_ratio=0.8
):
    """
    Prepare time series data for RNN training.

    Args:
        data: DataFrame with time series data
        sequence_length: Length of input sequences
        prediction_steps: How many steps ahead to predict
        train_ratio: Ratio of data to use for training

    Returns:
        X_train, y_train, X_val, y_val, scaler
    """
    # Extract values and reshape
    values = data["value"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []

    for i in range(len(scaled_values) - sequence_length - prediction_steps + 1):
        X.append(scaled_values[i : i + sequence_length])

        # For multi-step prediction
        if prediction_steps == 1:
            y.append(scaled_values[i + sequence_length])
        else:
            y.append(
                scaled_values[
                    i + sequence_length : i + sequence_length + prediction_steps
                ]
            )

    X = np.array(X)
    y = np.array(y)

    # Split into train and validation sets
    train_size = int(len(X) * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, y_train, X_val, y_val, scaler


def visualize_training_results(train_loss, val_loss, dataset_name):
    """
    Visualize training and validation loss.

    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        dataset_name: Name of the dataset for plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(f"Loss Curves for {dataset_name} Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_predictions(predictions, y_true, dataset_name, scaler=None):
    """
    Visualize model predictions against true values.

    Args:
        predictions: Model predictions (scaled)
        y_true: True values (scaled)
        dataset_name: Name of the dataset for plot title
        scaler: Scaler object to invert scaling
    """
    # Convert predictions and true values to original scale if scaler is provided
    if scaler:
        if predictions.shape[1:] != y_true.shape[1:]:
            # Handle different shapes between predictions and y_true
            if len(predictions.shape) == 2:
                predictions = predictions.reshape(-1, 1)
            if len(y_true.shape) == 2:
                y_true = y_true.reshape(-1, 1)

        predictions = scaler.inverse_transform(predictions)
        y_true = scaler.inverse_transform(y_true)

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, "b-", label="True Values")
    plt.plot(predictions, "r--", label="Predictions")
    plt.title(f"Predictions vs True Values for {dataset_name} Dataset")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_rnn_for_time_series(
    dataset_name, sequence_length=10, hidden_size=32, prediction_steps=1
):
    """
    Train an RNN for time series prediction.

    Args:
        dataset_name: Name of the dataset to use
        sequence_length: Length of input sequences
        hidden_size: Size of RNN hidden layer
        prediction_steps: How many steps ahead to predict

    Returns:
        rnn: Trained RNN model
        evaluation_metrics: Dictionary of evaluation metrics
    """
    # Fetch data
    datasets = fetch_data()

    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {list(datasets.keys())}"
        )

    data = datasets[dataset_name]

    # Prepare data
    X_train, y_train, X_val, y_val, scaler = prepare_time_series_data(
        data, sequence_length, prediction_steps, train_ratio=0.8
    )

    # Print data shapes
    print(f"Dataset: {dataset_name}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Create RNN
    input_size = 1
    output_size = prediction_steps

    rnn = RNN(input_size, hidden_size, output_size)

    # Train RNN
    print(f"\nTraining RNN on {dataset_name} dataset...")

    # Prepare data for RNN
    X_train_rnn = list(X_train)
    y_train_rnn = list(y_train)

    train_losses = []
    val_losses = []

    epochs = 100
    for epoch in range(epochs):
        # Train
        epoch_loss = 0
        for i in range(len(X_train_rnn)):
            outputs, _ = rnn.forward_sequence(X_train_rnn[i])
            loss = rnn.backward(y_train_rnn[i], learning_rate=0.01, clip_value=5.0)
            epoch_loss += loss

        avg_train_loss = epoch_loss / len(X_train_rnn)
        train_losses.append(avg_train_loss)

        # Validate
        val_loss = 0
        for i in range(len(X_val)):
            outputs = rnn.predict(X_val[i])

            # Calculate validation loss
            loss = 0
            for j in range(len(outputs)):
                if prediction_steps == 1:
                    # Single-step prediction
                    loss += np.sum((outputs[j] - y_val[i]) ** 2) / 2
                else:
                    # Multi-step prediction (comparing with multiple targets)
                    loss += np.sum((outputs[j] - y_val[i][j]) ** 2) / 2

            val_loss += loss / len(outputs)

        avg_val_loss = val_loss / len(X_val)
        val_losses.append(avg_val_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

    # Visualize training results
    visualize_training_results(train_losses, val_losses, dataset_name)

    # Make predictions
    predictions = []
    for i in range(len(X_val)):
        outputs = rnn.predict(X_val[i])
        if prediction_steps == 1:
            # For single-step prediction, use the last output
            predictions.append(outputs[-1])
        else:
            # For multi-step prediction, collect all outputs
            predictions.append(np.array([output for output in outputs]))

    predictions = np.array(predictions)

    # Reshape predictions if needed
    if prediction_steps == 1:
        predictions = np.array([p[0] for p in predictions])

    # Evaluate
    if prediction_steps == 1:
        mse = mean_squared_error(y_val, predictions)
        mae = mean_absolute_error(y_val, predictions)
    else:
        # For multi-step, flatten for evaluation
        y_val_flat = y_val.reshape(-1, 1)
        predictions_flat = predictions.reshape(-1, 1)
        mse = mean_squared_error(y_val_flat, predictions_flat)
        mae = mean_absolute_error(y_val_flat, predictions_flat)

    print(f"\nEvaluation metrics for {dataset_name} dataset:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    # Visualize predictions
    if prediction_steps == 1:
        # Reshape for visualization
        visualize_predictions(predictions, y_val, dataset_name, scaler)
    else:
        # For multi-step, visualize the first few sequences
        for i in range(min(3, len(predictions))):
            sample_pred = predictions[i]
            sample_true = y_val[i]
            visualize_predictions(
                sample_pred, sample_true, f"{dataset_name} (Sample {i+1})", scaler
            )

    return rnn, {"mse": mse, "mae": mae}


def compare_sequence_lengths(dataset_name, sequence_lengths=[5, 10, 20, 30]):
    """
    Compare RNN performance with different input sequence lengths.

    Args:
        dataset_name: Name of the dataset to use
        sequence_lengths: List of sequence lengths to compare
    """
    print(f"\nComparing different sequence lengths on {dataset_name} dataset...")

    results = []

    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        _, metrics = train_rnn_for_time_series(
            dataset_name, sequence_length=seq_len, hidden_size=32, prediction_steps=1
        )

        results.append((seq_len, metrics["mse"], metrics["mae"]))

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([r[0] for r in results], [r[1] for r in results], "o-")
    plt.title("MSE vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r[0] for r in results], [r[2] for r in results], "o-")
    plt.title("MAE vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary of sequence length comparison:")
    print(f"{'Sequence Length':<15} {'MSE':<10} {'MAE':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r[0]:<15} {r[1]:<10.6f} {r[2]:<10.6f}")


def explore_multi_step_prediction(
    dataset_name, sequence_length=15, prediction_steps_list=[1, 3, 5, 7]
):
    """
    Explore multi-step prediction with RNNs.

    Args:
        dataset_name: Name of the dataset to use
        sequence_length: Length of input sequences
        prediction_steps_list: List of prediction horizons to test
    """
    print(f"\nExploring multi-step prediction on {dataset_name} dataset...")

    results = []

    for pred_steps in prediction_steps_list:
        print(f"\nTesting {pred_steps}-step ahead prediction:")

        _, metrics = train_rnn_for_time_series(
            dataset_name,
            sequence_length=sequence_length,
            hidden_size=32,
            prediction_steps=pred_steps,
        )

        results.append((pred_steps, metrics["mse"], metrics["mae"]))

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([r[0] for r in results], [r[1] for r in results], "o-")
    plt.title("MSE vs Prediction Horizon")
    plt.xlabel("Steps Ahead")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r[0] for r in results], [r[2] for r in results], "o-")
    plt.title("MAE vs Prediction Horizon")
    plt.xlabel("Steps Ahead")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary of multi-step prediction comparison:")
    print(f"{'Steps Ahead':<15} {'MSE':<10} {'MAE':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r[0]:<15} {r[1]:<10.6f} {r[2]:<10.6f}")


def visualize_dataset(data, name):
    """
    Visualize a time series dataset.

    Args:
        data: DataFrame with time series data
        name: Name of the dataset for the title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["value"])
    plt.title(f"Time Series Data: {name}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the time series experiments."""
    print("===== Time Series Prediction with RNNs Exercise =====")

    # Fetch all datasets
    print("\nFetching datasets...")
    datasets = fetch_data()

    # Visualize datasets
    print("\nVisualizing datasets:")
    for name, data in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {data.shape}")
        print(f"Time range: {data.index.min()} to {data.index.max()}")
        print(f"Sample values: {data['value'].describe()}")

        visualize_dataset(data, name)

    # 1. Basic RNN forecasting on synthetic data
    print("\n1. Basic RNN forecasting on synthetic data")
    train_rnn_for_time_series(
        "synthetic", sequence_length=10, hidden_size=32, prediction_steps=1
    )

    # 2. Compare different sequence lengths
    print("\n2. Comparing different sequence lengths")
    compare_sequence_lengths("synthetic", sequence_lengths=[5, 10, 20, 30])

    # 3. Real-world time series: Air Passengers
    print("\n3. RNN forecasting on Air Passengers dataset")
    train_rnn_for_time_series(
        "air_passengers", sequence_length=12, hidden_size=48, prediction_steps=1
    )

    # 4. Multi-step prediction
    print("\n4. Multi-step prediction")
    explore_multi_step_prediction(
        "synthetic", sequence_length=15, prediction_steps_list=[1, 3, 5, 7]
    )

    print("\nThis exercise demonstrated:")
    print("- How to prepare time series data for RNN training")
    print("- Applying RNNs to both synthetic and real-world time series")
    print("- The impact of sequence length on prediction accuracy")
    print("- Multi-step ahead forecasting techniques")


if __name__ == "__main__":
    main()
