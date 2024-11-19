import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import Callback

# Configurable parameters
MAX_FEATURES = 10000  # Vocabulary size
MAX_LEN = 500  # Maximum length for long sequences
SHORT_MAX_LEN = 50  # Maximum length for short sequences
EPOCHS = 5  # Number of epochs for training
BATCH_SIZE = 32  # Batch size for training

# Load the IMDB dataset
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
print(f"Loaded {len(x_train)} training samples and {len(x_test)} testing samples.")

# Function to filter data based on sequence length
def filter_by_length(data, labels, max_length=None, min_length=None):
    filtered_data = []
    filtered_labels = []
    for i, sequence in enumerate(data):
        if max_length is not None and len(sequence) > max_length:
            continue
        if min_length is not None and len(sequence) < min_length:
            continue
        filtered_data.append(sequence)
        filtered_labels.append(labels[i])
    return np.array(filtered_data), np.array(filtered_labels)

# Separate datasets
x_train_short, y_train_short = filter_by_length(x_train, y_train, max_length=SHORT_MAX_LEN)
x_test_short, y_test_short = filter_by_length(x_test, y_test, max_length=SHORT_MAX_LEN)

x_train_long, y_train_long = filter_by_length(x_train, y_train, min_length=SHORT_MAX_LEN + 1, max_length=MAX_LEN)
x_test_long, y_test_long = filter_by_length(x_test, y_test, min_length=SHORT_MAX_LEN + 1, max_length=MAX_LEN)

# Mixed datasets (full data, no length filtering)
x_train_mixed, y_train_mixed = x_train, y_train
x_test_mixed, y_test_mixed = x_test, y_test

# Pad sequences
x_train_short = sequence.pad_sequences(x_train_short, maxlen=SHORT_MAX_LEN)
x_test_short = sequence.pad_sequences(x_test_short, maxlen=SHORT_MAX_LEN)

x_train_long = sequence.pad_sequences(x_train_long, maxlen=MAX_LEN)
x_test_long = sequence.pad_sequences(x_test_long, maxlen=MAX_LEN)

x_train_mixed = sequence.pad_sequences(x_train_mixed, maxlen=MAX_LEN)
x_test_mixed = sequence.pad_sequences(x_test_mixed, maxlen=MAX_LEN)

print(f"Short sequences: {len(x_train_short)} training samples.")
print(f"Long sequences: {len(x_train_long)} training samples.")
print(f"Mixed sequences: {len(x_train_mixed)} training samples.")


# Custom callback to monitor gradients using GradientTape
class GradientMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        with tf.GradientTape() as tape:
            predictions = self.model(x_val, training=False)
            loss = self.model.compiled_loss(y_val, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        avg_grad = np.mean([np.mean(np.abs(grad)) for grad in gradients if grad is not None])
        print(f"Epoch {epoch + 1} - Average Gradient Magnitude: {avg_grad:.6f}")


# Function to build a simple RNN model
def build_rnn_model(input_length):
    model = Sequential([
        Embedding(MAX_FEATURES, 32, input_length=input_length),
        SimpleRNN(16),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Training function
def train_and_evaluate(x_train, y_train, x_test, y_test, input_length, title):
    print(f"\nTraining RNN on {title} dataset...")
    model = build_rnn_model(input_length)
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[GradientMonitor()]
    )
    return history


# Train and compare models
history_short = train_and_evaluate(x_train_short, y_train_short, x_test_short, y_test_short, SHORT_MAX_LEN, "short sequences")
history_long = train_and_evaluate(x_train_long, y_train_long, x_test_long, y_test_long, MAX_LEN, "long sequences")
history_mixed = train_and_evaluate(x_train_mixed, y_train_mixed, x_test_mixed, y_test_mixed, MAX_LEN, "mixed sequences")


# Plot training histories
def plot_training_histories(histories, titles):
    plt.figure(figsize=(18, 12))
    for i, (history, title) in enumerate(zip(histories, titles)):
        plt.subplot(3, 2, i * 2 + 1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title(f'{title} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(3, 2, i * 2 + 2)
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_histories(
    [history_short, history_long, history_mixed],
    ["Short Sequences", "Long Sequences", "Mixed Sequences"]
)
