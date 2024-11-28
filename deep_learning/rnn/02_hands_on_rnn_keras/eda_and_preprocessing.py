import logging
import os

import numpy as np
from plotly import graph_objects as go
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from config import Config

# Get the existing logger configured in main.py
logger = logging.getLogger()


def load_and_preprocess_data(min_length: int, max_length: int, max_features: int) -> tuple:
    """Loads and preprocesses the IMDB dataset with splits for train, val, and test."""
    logger.info(
        f"Starting data loading and preprocessing with min_length={min_length}, max_length={max_length}, max_features={max_features}."
    )

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=Config.MAX_FEATURES)
    logger.info(f"Data loaded: train={len(x_train)}, test={len(x_test)}.")

    x_train, y_train = filter_by_length(x_train, y_train, min_length, max_length)
    x_test, y_test = filter_by_length(x_test, y_test, min_length, max_length)
    logger.info(
        f"Data filtered: train={len(x_train)}, test={len(x_test)} "
        f"with min_length={min_length}, max_length={max_length}."
    )

    unique, counts = np.unique(y_train, return_counts=True)
    unique, counts = unique.tolist(), counts.tolist()
    logger.info(f"Classes distribution: {dict(zip(unique, counts))}")

    logger.info("Splitting data into train, val, and test sets.")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.35, random_state=Config.RANDOM_SEED, stratify=y_train
    )

    if Config.DEV_MODE:
        logger.info(f"DEV_MODE active: Using {Config.DEV_SAMPLES} samples.")
        x_train, y_train = x_train[: Config.DEV_SAMPLES], y_train[: Config.DEV_SAMPLES]
        x_val, y_val = x_val[: Config.DEV_SAMPLES], y_val[: Config.DEV_SAMPLES]
        x_test, y_test = x_test[: Config.DEV_SAMPLES], y_test[: Config.DEV_SAMPLES]
        logger.info(
            f"DEV_MODE active: Using {Config.DEV_SAMPLES} samples: "
            f"train={len(x_train)}, val={len(x_val)}, test={len(x_test)}."
        )

    x_train, x_val, x_test = map(
        lambda x: truncate_indices(x, max_features), (x_train, x_val, x_test)
    )
    logger.info(f"Padding sequences to a maximum length of {Config.MAX_LEN}.")
    x_train, x_val, x_test = map(
        lambda x: sequence.pad_sequences(x, maxlen=Config.MAX_LEN), (x_train, x_val, x_test)
    )

    logger.info(
        f"Data loaded and preprocessed: train={len(x_train)}, "
        f"val={len(x_val)}, test={len(x_test)}."
    )

    return (
        (np.array(x_train), np.array(y_train)),
        (np.array(x_val), np.array(y_val)),
        (np.array(x_test), np.array(y_test)),
    )


def filter_by_length(data, labels, min_length, max_length):
    """Filters sequences by length."""
    logger.info(f"Filtering sequences by length: {min_length} <= length <= {max_length}.")
    filtered_data, filtered_labels = [], []
    for seq, label in zip(data, labels):
        if min_length <= len(seq) <= max_length:
            filtered_data.append(seq)
            filtered_labels.append(label)
    logger.info(
        f"Filtered data: {len(filtered_data)} sequences. Filtered labels: {len(filtered_labels)}"
    )
    return np.array(filtered_data, dtype=object), np.array(filtered_labels)


def truncate_indices(data, max_features):
    """Truncates indices in sequences to a maximum feature value."""
    logger.info(f"Truncating indices to a maximum of {max_features}.")
    return [[min(word, max_features - 1) for word in seq] for seq in data]


def get_statistics(data_lengths):
    """Compute summary statistics for sequence lengths."""
    logger.info("Computing summary statistics for sequence lengths.")
    mean = np.mean(data_lengths)
    median = np.median(data_lengths)
    max_len = np.max(data_lengths)
    q1 = np.percentile(data_lengths, 25)
    q2 = np.percentile(data_lengths, 50)
    q3 = np.percentile(data_lengths, 75)
    p95 = np.percentile(data_lengths, 95)
    p99 = np.percentile(data_lengths, 99)

    logger.info(
        f"Mean: {mean:.2f}, Median: {median:.2f}, Max: {max_len:.2f}\n"
        f"Q1 (25th percentile): {q1:.2f}, Q2 (Median): {q2:.2f}, "
        f"Q3 (75th percentile): {q3:.2f}\n95th Percentile: {p95:.2f}, "
        f"99th Percentile: {p99:.2f}"
    )

    return mean, median, max_len, q1, q2, q3, p95, p99


def print_statistics(mean, median, max_len, q1, q2, q3, p95, p99):
    """Print summary statistics for sequence lengths."""
    logger.info("Printing summary statistics for sequence lengths.")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Max: {max_len:.2f}")
    print(f"First Quartile (Q1 - 25th percentile): {q1:.2f}")
    print(f"Second Quartile (Q2 - 50th percentile - Median): {q2:.2f}")
    print(f"Third Quartile (Q3 - 75th percentile): {q3:.2f}")
    print(f"95th Percentile: {p95:.2f}")
    print(f"99th Percentile: {p99:.2f}")


def save_hist_and_quartiles_plotly(
    data_lengths,
    q1,
    q2,
    q3,
    p95,
    p99,
    save_filename: str = "sequence_lengths.html",
    save_dir: str = os.path.join(os.getcwd(), "eda", "plots"),
):
    """Plot histogram of sequence lengths with quartiles using Plotly."""
    logger.info("Saving histogram of sequence lengths with quartiles using Plotly.")
    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=data_lengths, nbinsx=50, marker_color="skyblue", opacity=0.7, name="Sequence Lengths"
        )
    )

    # Add vertical lines for quartiles with hover tooltips
    for value, label, color in zip(
        [q1, q2, q3, p95, p99],
        [
            "Q1 (25th percentile)",
            "Q2 (Median)",
            "Q3 (75th percentile)",
            "95th Percentile",
            "99th Percentile",
        ],
        ["blue", "green", "red", "purple", "orange"],
    ):
        fig.add_shape(
            type="line",
            x0=value,
            y0=0,
            x1=value,
            y1=1,
            line=dict(color=color, dash="dash"),
            xref="x",
            yref="paper",
        )
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[0],
                mode="markers",
                marker=dict(size=5, color=color),
                name=f"{label}: {value:0.0f}",
                hovertemplate=f"{label}: <b>{value:0.0f}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title="Sequence Length Distribution with Quartiles",
        xaxis_title="Sequence Length",
        yaxis_title="Frequency",
        template="plotly_white",
        bargap=0.1,
        height=600,  # Increase height to give space for annotations
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    fig.write_html(save_path)
    logger.info(f"Plot saved to {save_path}")


def compute_length_buckets(train_lengths, test_lengths):
    """
    Compute dynamic length buckets based on dataset statistics.

    Args:
        train_lengths: List or array of sequence lengths from the training set.
        test_lengths: List or array of sequence lengths from the test set.

    Returns:
        A list of tuples representing dynamic length buckets.
    """
    logger.info("Computing dynamic length buckets based on dataset statistics.")
    # Combine train and test lengths into a single dataset
    combined_lengths = np.concatenate([train_lengths, test_lengths])

    # Compute summary statistics dynamically
    _, _, max_len, q1, q2, q3, p95, p99 = get_statistics(combined_lengths)

    # Create dynamic length buckets based on these statistics
    return [
        (0, int(q1)),  # 0 to Q1
        (int(q1), int(q2) - 1),  # Q1 to Median
        (int(q2), int(q3) - 1),  # Median to Q3
        (int(q3), int(p95) - 1),  # Q3 to 95th Percentile
        (int(p95), int(max_len)),  # 95th Percentile to Max
    ]


def load_dataset_compute_length_buckets():
    """
    Load the IMDB dataset and compute dynamic length buckets.
    :return: A list of tuples representing dynamic length buckets.
    """
    logger.info("Loading IMDB dataset and computing dynamic length buckets.")
    # Load data with default settings
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    # Compute sequence lengths for training and test sets
    train_lengths = [len(seq) for seq in x_train]
    test_lengths = [len(seq) for seq in x_test]

    # Compute statistics for training and test sets
    train_mean, train_median, train_max, train_q1, train_q2, train_q3, train_p95, train_p99 = (
        get_statistics(train_lengths)
    )
    test_mean, test_median, test_max, test_q1, test_q2, test_q3, test_p95, test_p99 = (
        get_statistics(test_lengths)
    )

    # Print statistics
    print("Train Lengths Statistics:")
    print_statistics(
        train_mean, train_median, train_max, train_q1, train_q2, train_q3, train_p95, train_p99
    )
    print("\nTest Lengths Statistics:")
    print_statistics(
        test_mean, test_median, test_max, test_q1, test_q2, test_q3, test_p95, test_p99
    )

    # Compute dynamic buckets
    length_buckets = compute_length_buckets(train_lengths, test_lengths)
    print("\nNumber of Buckets:", len(length_buckets))

    # Print the number of sequences in each bucket
    for i, (start, end) in enumerate(length_buckets):
        bucket_count = sum(1 for length in train_lengths if start <= length < end)
        print(f"Bucket {i + 1}: {bucket_count} sequences")

    print("Dynamic Length Buckets:", length_buckets)

    save_hist_and_quartiles_plotly(
        train_lengths,
        train_q1,
        train_q2,
        train_q3,
        train_p95,
        train_p99,
        save_filename="train_sequence_lengths.html",
    )

    save_hist_and_quartiles_plotly(
        test_lengths,
        test_q1,
        test_q2,
        test_q3,
        test_p95,
        test_p99,
        save_filename="test_sequence_lengths.html",
    )

    return length_buckets


def main():
    """
    Main function to automate experiments across length buckets.
    """
    load_dataset_compute_length_buckets()


if __name__ == "__main__":
    main()
