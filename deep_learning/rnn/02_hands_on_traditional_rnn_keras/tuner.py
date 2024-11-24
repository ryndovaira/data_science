import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.models import Sequential

from config import Config
from data_preprocessing import load_and_preprocess_data
from utils import checkpoint_path, get_artifacts_dir
from logger import setup_logger
from plotter import plot_history
from saver import save_model, save_history, save_tuner_results

# Setup logger
logger = setup_logger()


def configure_tf_device():
    """Configure TensorFlow to use GPU if available."""
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
    embedding_dim = hp.Choice("embedding_dim", values=[32, 64, 128])
    rnn_units = hp.Int("rnn_units", min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)

    model = Sequential(
        [
            Embedding(input_dim=Config.MAX_FEATURES, output_dim=embedding_dim),
            SimpleRNN(rnn_units),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def tune_hyperparameters():
    """Tunes hyperparameters using Keras Tuner and generates trial plots."""
    configure_tf_device()  # Ensure TensorFlow is set up for GPUs
    logger.info("Starting hyperparameter tuning.")

    # Pass dummy max features for initial data loading (updated dynamically during tuning)
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess_data(
        Config.MIN_LEN, Config.MAX_LEN, Config.MAX_FEATURES
    )

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=Config.TUNER_MAX_EPOCHS,
        factor=Config.HYPERBAND_FACTOR,
        directory=get_artifacts_dir(Config.TUNER_DIR),
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
            callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
        )
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"Best hyperparameters found: {best_hps.values}")

    save_tuner_results(tuner)

    return best_hps


def retrain_with_best_hps(best_hps):
    """Retrains the model using the best hyperparameters."""
    configure_tf_device()  # Ensure TensorFlow is set up for GPUs
    logger.info("Retraining model with the best hyperparameters.")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
        Config.MIN_LEN, Config.MAX_LEN, Config.MAX_FEATURES
    )

    model = model_builder(best_hps)
    history = model.fit(
        x_train,
        y_train,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path(), monitor="val_loss", save_best_only=True),
        ],
    )

    save_model(model)
    plot_history(history)
    save_history(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy
