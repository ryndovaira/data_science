# main.py
from config import Config
from data_preprocessing import load_and_preprocess_data
from model import build_rnn_model
from utils import setup_logger, save_artifacts
from callbacks import GradientMonitor
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def main():
    # Setup logger
    logger = setup_logger('training')

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    logger.info("Data loaded and preprocessed.")

    # Build and compile model
    model = build_rnn_model(Config.MAX_LEN)
    logger.info("Model built.")

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[
            GradientMonitor(),
            EarlyStopping(patience=2, restore_best_weights=True),
            ModelCheckpoint(filepath='artifacts/best_model.keras', save_best_only=True)
        ]
    )
    logger.info("Model training completed.")

    # Save artifacts and results
    save_artifacts(history, model, 'artifacts/')
    logger.info("Artifacts saved.")

if __name__ == "__main__":
    main()
