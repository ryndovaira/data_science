import logging

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from config import Config

# Get the existing logger configured in main.py
logger = logging.getLogger()


def load_and_preprocess_data(min_length: int, max_length: int) -> tuple:
    """Loads and preprocesses the IMDB dataset with splits for train, val, and test based on the selected mode in Config."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=Config.MAX_FEATURES)

    # Apply mode-specific filtering first
    x_train, y_train = filter_by_length(
        x_train, y_train, min_length=min_length, max_length=max_length
    )
    x_test, y_test = filter_by_length(x_test, y_test, min_length=min_length, max_length=max_length)
    logger.info(f"Data loaded and filtered by length [{min_length}; {max_length})")

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=Config.RANDOM_SEED
    )

    # If in dev mode, reduce the dataset size after filtering
    if Config.DEV_MODE:
        x_train = x_train[: Config.DEV_SAMPLES]
        y_train = y_train[: Config.DEV_SAMPLES]
        x_val = x_val[: Config.DEV_SAMPLES]
        y_val = y_val[: Config.DEV_SAMPLES]
        x_test = x_test[: Config.DEV_SAMPLES]
        y_test = y_test[: Config.DEV_SAMPLES]
        logger.info(
            f"Running in dev mode with {Config.DEV_SAMPLES} samples for training, validation, and testing."
        )

    # Pad sequences
    x_train = np.array(sequence.pad_sequences(x_train, maxlen=max_length))
    x_val = np.array(sequence.pad_sequences(x_val, maxlen=max_length))
    x_test = np.array(sequence.pad_sequences(x_test, maxlen=max_length))

    return (x_train, np.array(y_train)), (x_val, np.array(y_val)), (x_test, np.array(y_test))


def filter_by_length(data, labels, min_length, max_length):
    """Filters sequences by length, with optional max and min lengths."""
    filtered_data = []
    filtered_labels = []
    for i, seq in enumerate(data):
        if len(seq) > max_length or len(seq) < min_length:
            continue
        filtered_data.append(seq)
        filtered_labels.append(labels[i])
    return filtered_data, np.array(filtered_labels)
