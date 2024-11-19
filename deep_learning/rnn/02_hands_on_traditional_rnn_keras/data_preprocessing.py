import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from config import Config


def load_and_preprocess_data():
    # Load data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=Config.MAX_FEATURES)

    # Pad sequences
    x_train = sequence.pad_sequences(x_train, maxlen=Config.MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=Config.MAX_LEN)

    return (x_train, y_train), (x_test, y_test)
