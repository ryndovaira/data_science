import logging

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from config import Config
from plotter import plot_hist_and_quartiles

# Get the existing logger configured in main.py
logger = logging.getLogger()


def load_dataset():
    logger.info("Loading IMDB dataset.")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=Config.MAX_FEATURES)
    logger.info(f"Data loaded: train={len(x_train)}, test={len(x_test)}.")
    return x_train, y_train, x_test, y_test


def preprocess_data(
    x_train, y_train, x_test, y_test, min_length: int, max_length: int, max_features: int
) -> tuple:
    """Preprocesses the IMDB dataset with splits for train, val, and test."""
    logger.info(
        f"Starting preprocessing with min_length={min_length}, max_length={max_length}, max_features={max_features}."
    )

    x_train, y_train = filter_by_length(x_train, y_train, min_length, max_length)
    x_test, y_test = filter_by_length(x_test, y_test, min_length, max_length)
    logger.info(
        f"Data filtered: train={len(x_train)}, test={len(x_test)} "
        f"with min_length={min_length}, max_length={max_length}."
    )

    class_distribution(y_train)

    logger.info("Splitting data into train, val, and test sets.")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.35, random_state=Config.RANDOM_SEED, stratify=y_train
    )

    if Config.DEV_MODE:
        x_test, x_train, x_val, y_test, y_train, y_val = dev_mode(
            x_test, x_train, x_val, y_test, y_train, y_val
        )

    logger.info(f"Truncating indices to a maximum of {max_features}.")
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


def dev_mode(x_test, x_train, x_val, y_test, y_train, y_val):
    logger.info(f"DEV_MODE active: Using {Config.DEV_SAMPLES} samples.")
    x_train, y_train = x_train[: Config.DEV_SAMPLES], y_train[: Config.DEV_SAMPLES]
    x_val, y_val = x_val[: Config.DEV_SAMPLES], y_val[: Config.DEV_SAMPLES]
    x_test, y_test = x_test[: Config.DEV_SAMPLES], y_test[: Config.DEV_SAMPLES]
    logger.info(
        f"DEV_MODE active: Using {Config.DEV_SAMPLES} samples: "
        f"train={len(x_train)}, val={len(x_val)}, test={len(x_test)}."
    )
    return x_test, x_train, x_val, y_test, y_train, y_val


def class_distribution(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    unique, counts = unique.tolist(), counts.tolist()
    logger.info(f"Classes distribution: {dict(zip(unique, counts))}")


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
        f"Mean: {mean}, Median: {median}, Max: {max_len}, Q1: {q1}, Q2: {q2}, Q3: {q3}, P95: {p95}, P99: {p99}"
    )

    return mean, median, max_len, q1, q2, q3, p95, p99


def compute_length_buckets(x_train, x_test):
    """
    Load the IMDB dataset and compute dynamic length buckets.
    :return: A list of tuples representing dynamic length buckets.
    """

    logger.info("Computing sequence lengths for training and test sets.")
    train_lengths = [len(seq) for seq in x_train]
    test_lengths = [len(seq) for seq in x_test]

    # Compute statistics for training and test sets
    train_mean, train_median, train_max, train_q1, train_q2, train_q3, train_p95, train_p99 = (
        get_statistics(train_lengths)
    )
    test_mean, test_median, test_max, test_q1, test_q2, test_q3, test_p95, test_p99 = (
        get_statistics(test_lengths)
    )

    logger.info("Computing dynamic length buckets based on dataset statistics.")
    combined_lengths = np.concatenate([train_lengths, test_lengths])
    _, _, max_len, q1, q2, q3, p95, p99 = get_statistics(combined_lengths)
    length_buckets = [
        (0, int(q1)),  # 0 to Q1
        (int(q1), int(q2) - 1),  # Q1 to Median
        (int(q2), int(q3) - 1),  # Median to Q3
        (int(q3), int(p95) - 1),  # Q3 to 95th Percentile
        (int(p95), int(max_len)),  # 95th Percentile to Max
    ]
    logger.info(f"Number of Buckets: {len(length_buckets)}")

    for i, (start, end) in enumerate(length_buckets):
        bucket_count = sum(1 for length in train_lengths if start <= length < end)
        logger.info(f"Bucket {i + 1}: {bucket_count} sequences")

    logger.info("Dynamic Length Buckets:", length_buckets)

    plot_hist_and_quartiles(
        train_lengths,
        train_q1,
        train_q2,
        train_q3,
        train_p95,
        train_p99,
        save_filename="train_sequence_lengths.html",
    )

    plot_hist_and_quartiles(
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
    Main function to load the IMDB dataset and compute dynamic length buckets.
    """
    x_train, y_train, x_test, y_test = load_dataset()
    compute_length_buckets(x_train, x_test)


if __name__ == "__main__":
    main()
