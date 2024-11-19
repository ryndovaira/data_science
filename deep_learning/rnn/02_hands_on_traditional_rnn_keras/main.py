from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import Config
from data_preprocessing import load_and_preprocess_data
from model import build_rnn_model
from utils import setup_logger, save_artifacts, checkpoint_path, plot_history, save_history


def train_model(model, x_train, y_train, x_val, y_val):
    """Trains the model with early stopping and model checkpointing."""
    return model.fit(
        x_train,
        y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[
            EarlyStopping(patience=2, restore_best_weights=True),
            ModelCheckpoint(
                filepath=checkpoint_path(),
                monitor="val_loss",
                save_best_only=False,
                verbose=1,
            ),
        ],
    )


def log_performance(logger, history, test_loss, test_accuracy):
    """Logs the training, validation, and test performance."""
    logger.info(
        f"Training Loss: {history.history['loss'][-1]:.4f}, Training Accuracy: {history.history['accuracy'][-1]:.4f}"
    )
    logger.info(
        f"Validation Loss: {history.history['val_loss'][-1]:.4f}, Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}"
    )
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def evaluate_model(model, x_test, y_test):
    """Evaluates the model on the test data."""
    return model.evaluate(x_test, y_test, verbose=1)


def main():
    # Setup logger
    logger = setup_logger()

    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    logger.info("Data loaded and preprocessed.")

    # Build and compile model
    model = build_rnn_model(Config.MAX_LEN)
    logger.info("Model built.")

    # Train the model
    history = train_model(model, x_train, y_train, x_val, y_val)
    logger.info("Model training completed.")

    # Save artifacts and results
    save_artifacts(history, model, "artifacts/")
    logger.info("Artifacts saved.")

    # Evaluate the model on the test data
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)

    # Log training, validation, and test performance
    log_performance(logger, history, test_loss, test_accuracy)

    # Update history to include test performance for plotting
    history.history["test_loss"] = [test_loss]
    history.history["test_accuracy"] = [test_accuracy]

    # Save history as a plot
    plot_history(history)

    # Save history as a JSON file
    save_history(history)


if __name__ == "__main__":
    main()
