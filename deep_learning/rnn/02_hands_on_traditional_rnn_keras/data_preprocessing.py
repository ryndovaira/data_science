import logging
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from config import Config

# Get the existing logger configured in main.py
logger = logging.getLogger()


def load_and_preprocess_data(min_length: int, max_length: int, max_features: int) -> tuple:
    """Loads and preprocesses the IMDB dataset with splits for train, val, and test."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=Config.MAX_FEATURES)

    logger.info(f"Filtering and preprocessing data with max_features={max_features}.")
    x_train, y_train = filter_by_length(x_train, y_train, min_length, max_length)
    x_test, y_test = filter_by_length(x_test, y_test, min_length, max_length)

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=Config.RANDOM_SEED
    )

    if Config.DEV_MODE:
        x_train, y_train = x_train[: Config.DEV_SAMPLES], y_train[: Config.DEV_SAMPLES]
        x_val, y_val = x_val[: Config.DEV_SAMPLES], y_val[: Config.DEV_SAMPLES]
        x_test, y_test = x_test[: Config.DEV_SAMPLES], y_test[: Config.DEV_SAMPLES]
        logger.info("Development mode active, reduced dataset size.")

    x_train, x_val, x_test = map(
        lambda x: truncate_indices(x, max_features), (x_train, x_val, x_test)
    )

    x_train, x_val, x_test = map(
        lambda x: sequence.pad_sequences(x, maxlen=Config.MAX_LEN), (x_train, x_val, x_test)
    )

    return (
        (np.array(x_train), np.array(y_train)),
        (np.array(x_val), np.array(y_val)),
        (np.array(x_test), np.array(y_test)),
    )


def filter_by_length(data, labels, min_length, max_length):
    """Filters sequences by length."""
    filtered_data, filtered_labels = [], []
    for seq, label in zip(data, labels):
        if min_length <= len(seq) <= max_length:
            filtered_data.append(seq)
            filtered_labels.append(label)
    return np.array(filtered_data, dtype=object), np.array(filtered_labels)


def truncate_indices(data, max_features):
    """Truncates indices in sequences to a maximum feature value."""
    return [[min(word, max_features - 1) for word in seq] for seq in data]
