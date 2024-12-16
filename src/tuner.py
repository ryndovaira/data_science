import keras_tuner as kt
import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN, Embedding, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.config import Config
from utils import checkpoint_path, get_artifacts_arch_dir
from src.logger import setup_logger
from src.plotter import plot_history
from saver import save_model, save_history, save_tuner_results

# Setup logger
logger = setup_logger()


def configure_tf_device():
    """Configure TensorFlow to use GPU if available."""
    logger.info("Configuring TensorFlow device.")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPUs detected and configured: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
    else:
        logger.info("No GPU detected, using CPU.")


def model_builder(hp):
    """Builds the model for hyperparameter tuning."""
    logger.info("Building model for hyperparameter tuning.")
    embedding_dim = hp.Choice("embedding_dim", values=[32, 64, 128, 256])
    rnn_units = hp.Int("rnn_units", min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)

    logger.info(f"Using {Config.ARCHITECTURE} architecture.")
    logger.info(f"Embedding Dimension: {embedding_dim}")
    logger.info(f"RNN Units: {rnn_units}")
    logger.info(f"Dropout Rate: {dropout_rate}")
    logger.info(f"Max Features: {Config.MAX_FEATURES}")

    model = Sequential()
    model.add(Embedding(input_dim=Config.MAX_FEATURES, output_dim=embedding_dim))

    # TODO: add layer stacking
    if Config.ARCHITECTURE == "VanillaRNN":
        model.add(SimpleRNN(rnn_units))
    elif Config.ARCHITECTURE == "LSTM":
        model.add(LSTM(rnn_units))
    elif Config.ARCHITECTURE == "GRU":
        model.add(GRU(rnn_units))
    else:
        raise ValueError(f"Unsupported architecture: {Config.ARCHITECTURE}")

    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
    logger.info("Model compiled.")
    return model


def tune_hyperparameters(x_train, y_train, x_val, y_val):
    """Tunes hyperparameters using Keras Tuner and generates trial plots."""
    logger.info("Starting hyperparameter tuning.")

    configure_tf_device()

    logger.info(f"Tuner Max Epochs: {Config.TUNER_MAX_EPOCHS}")
    logger.info(f"Hyperband Factor: {Config.HYPERBAND_FACTOR}")
    logger.info(f"Hyperband Iterations: {Config.HYPERBAND_ITERATIONS}")
    logger.info(f"Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"Patience: {Config.PATIENCE}")

    logger.info("Starting hyperparameter tuning.")

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=Config.TUNER_MAX_EPOCHS,
        factor=Config.HYPERBAND_FACTOR,
        directory=get_artifacts_arch_dir(Config.TUNER_DIR),
        project_name=f"{Config.HYPERBAND_PROJ_NAME}_{Config.min_max_len()}",
        hyperband_iterations=Config.HYPERBAND_ITERATIONS,
    )
    try:
        tuner.search(
            x_train,
            y_train,
            epochs=Config.TUNER_MAX_EPOCHS,
            validation_data=(x_val, y_val),
            batch_size=Config.BATCH_SIZE,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=Config.PATIENCE),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.2, patience=Config.PATIENCE, min_lr=1e-6, verbose=1
                ),
            ],
        )
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise
    logger.info("Hyperparameter tuning completed.")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"Best hyperparameters found: {best_hps.values}")

    save_tuner_results(tuner)

    return best_hps


def retrain_with_best_hps(best_hps, x_train, y_train, x_val, y_val, x_test, y_test):
    """Retrains the model using the best hyperparameters."""
    logger.info("Retraining model with the best hyperparameters.")

    configure_tf_device()

    logger.info(f"Epochs: {Config.EPOCHS}")
    logger.info(f"Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"Patience: {Config.PATIENCE}")

    model = model_builder(best_hps)

    logger.info("Model training started.")
    history = model.fit(
        x_train,
        y_train,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=Config.PATIENCE, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path(), monitor="val_loss", save_best_only=True),
        ],
    )
    logger.info("Model training completed.")

    save_model(model)
    plot_history(history)
    save_history(history)

    logger.info("Model evaluation on a test set.")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    logger.info(f"Final loss and accuracy:")
    train_loss = history.history["loss"][-1]
    train_accuracy = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy
