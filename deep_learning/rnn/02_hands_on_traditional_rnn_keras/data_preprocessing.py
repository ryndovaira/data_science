import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from config import Config


def load_and_preprocess_data():
    """Loads and preprocesses the IMDB dataset with splits for train, val, and test based on the selected mode in Config."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=Config.MAX_FEATURES)

    # Apply mode-specific filtering first
    if Config.MODE == "short":
        x_train, y_train = filter_by_length(x_train, y_train, max_length=Config.SHORT_MAX_LEN)
        x_test, y_test = filter_by_length(x_test, y_test, max_length=Config.SHORT_MAX_LEN)
        max_len = Config.SHORT_MAX_LEN
        print("Running in short sequence mode.")
    elif Config.MODE == "long":
        x_train, y_train = filter_by_length(
            x_train, y_train, min_length=Config.SHORT_MAX_LEN + 1, max_length=Config.MAX_LEN
        )
        x_test, y_test = filter_by_length(
            x_test, y_test, min_length=Config.SHORT_MAX_LEN + 1, max_length=Config.MAX_LEN
        )
        max_len = Config.MAX_LEN
        print("Running in long sequence mode.")
    else:  # mixed mode
        max_len = Config.MAX_LEN
        print("Running in mixed sequence mode.")

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # If in dev mode, reduce the dataset size after filtering
    if Config.DEV_MODE:
        x_train = x_train[: Config.DEV_SAMPLES]
        y_train = y_train[: Config.DEV_SAMPLES]
        x_val = x_val[: Config.DEV_SAMPLES]
        y_val = y_val[: Config.DEV_SAMPLES]
        x_test = x_test[: Config.DEV_SAMPLES]
        y_test = y_test[: Config.DEV_SAMPLES]
        print(
            f"Running in dev mode with {Config.DEV_SAMPLES} samples for training, validation, and testing."
        )

    # Pad sequences
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_val = sequence.pad_sequences(x_val, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def filter_by_length(data, labels, max_length=None, min_length=None):
    """Filters sequences by length, with optional max and min lengths."""
    filtered_data = []
    filtered_labels = []
    for i, seq in enumerate(data):
        if max_length is not None and len(seq) > max_length:
            continue
        if min_length is not None and len(seq) < min_length:
            continue
        filtered_data.append(seq)
        filtered_labels.append(labels[i])
    return np.array(filtered_data), np.array(filtered_labels)
