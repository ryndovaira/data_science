from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from config import Config


def build_rnn_model():
    model = Sequential(
        [
            Embedding(Config.MAX_FEATURES, Config.EMBEDDING_DIM),
            SimpleRNN(Config.RNN_UNITS),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
