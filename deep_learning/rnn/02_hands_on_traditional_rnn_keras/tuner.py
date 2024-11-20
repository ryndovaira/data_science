import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from data_preprocessing import load_and_preprocess_data
from config import Config
from utils import setup_logger, save_model, checkpoint_path, plot_history

# Setup logger
logger = setup_logger()


def model_builder(hp):
    """Builds the model for hyperparameter tuning."""
    embedding_dim = hp.Choice("embedding_dim", values=[32, 64, 128])
    rnn_units = hp.Int("rnn_units", min_value=16, max_value=128, step=16)
    max_features = hp.Choice("max_features", values=[5000, 10000, 20000])

    model = Sequential(
        [
            Embedding(
                input_dim=max_features, output_dim=embedding_dim
            ),
            SimpleRNN(rnn_units),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def tune_hyperparameters():
    """Tunes hyperparameters using Keras Tuner."""
    logger.info("Starting hyperparameter tuning.")

    # Pass dummy max_features for initial data loading (updated dynamically during tuning)
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess_data(
        Config.MIN_LEN, Config.MAX_LEN, 5000
    )

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=Config.TUNER_MAX_EPOCHS,
        factor=3,
        directory="tuner_results",
        project_name="imdb_rnn_tuning",
        hyperband_iterations=4,
    )

    tuner.search(
        x_train,
        y_train,
        epochs=Config.TUNER_MAX_EPOCHS,
        validation_data=(x_val, y_val),
        batch_size=Config.BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"Best hyperparameters found: {best_hps.values}")
    return best_hps


def retrain_with_best_hps(best_hps):
    """Retrains the model using the best hyperparameters."""
    logger.info("Retraining model with best hyperparameters.")

    # Use the best max_features for data preprocessing
    max_features = best_hps.get("max_features")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
        Config.MIN_LEN, Config.MAX_LEN, max_features
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

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
