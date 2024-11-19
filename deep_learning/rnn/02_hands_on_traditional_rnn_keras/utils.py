import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

from config import Config


def get_timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def setup_logger():
    """Sets up a logger that writes to a specified log file with a unique timestamp."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), Config.LOG_DIR, Config.MODE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add a timestamp to the log file name
    log_file_path = os.path.join(log_dir, f"log_{get_timestamp()}.log")

    # Create and configure the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers if the logger is reused
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def save_artifacts(history, model, output_dir):
    """Saves training history and model artifacts with unique timestamps."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model with a unique filename
    model_filename = f"model_{get_timestamp()}.keras"
    model.save(os.path.join(output_dir, Config.MODE, model_filename))


def checkpoint_path():
    # Generate a unique base filename with timestamp
    checkpoint_dir = os.path.join(Config.ARTIFACTS_DIR, Config.MODE)

    # Create output directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    s = f"model_checkpoint_{get_timestamp()}" + "_epoch-{epoch:02d}_val-loss-{val_loss:.4f}.keras"
    return os.path.join(checkpoint_dir, s)


def plot_history(history):
    """Plots and saves the training history as an image file."""
    # Create logs directory if it doesn't exist
    save_dir = os.path.join(os.getcwd(), Config.PLOT_DIR, Config.MODE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_path = os.path.join(save_dir, f"plot_{get_timestamp()}.png")

    plt.figure(figsize=(10, 6))
    # Plot training and validation loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    # If test data is available, plot it as a single point at the end
    if "test_loss" in history.history:
        plt.scatter(
            len(history.history["loss"]) - 1,
            history.history["test_loss"][0],
            color="red",
            label="Test Loss",
        )

    # Plot training and validation accuracy
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    # If test data is available, plot it as a single point at the end
    if "test_accuracy" in history.history:
        plt.scatter(
            len(history.history["accuracy"]) - 1,
            history.history["test_accuracy"][0],
            color="blue",
            label="Test Accuracy",
        )

    plt.title("Training, Validation, and Test Loss/Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_file_path)
    plt.close()
