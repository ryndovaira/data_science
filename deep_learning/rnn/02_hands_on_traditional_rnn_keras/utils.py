import json
import logging
import os

import matplotlib.pyplot as plt

from config import Config


# def get_dir_path(base_dir: str, *sub_dirs: str) -> str:
#     dir_path = os.path.join(os.getcwd(), base_dir, Config.name(), *sub_dirs)
#     os.makedirs(dir_path, exist_ok=True)
#     return dir_path


def get_artifacts_dir(base_dir: str, *sub_dirs: str) -> str:
    """Returns the directory path for saving artifacts, ensuring it exists."""
    dir_path = os.path.join(os.getcwd(), Config.ARTIFACTS_DIR, base_dir, Config.name(), *sub_dirs)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def setup_logger():
    """Sets up a logger that writes to a specified log file with a unique timestamp."""

    # Define the log directory path
    save_dir = get_artifacts_dir(Config.LOG_DIR)

    # Define the log file path with a timestamp
    log_file_path = os.path.join(save_dir, f"log_{Config.TIMESTAMP}.txt")

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


def save_model(model: "tf.keras.Model"):
    """Saves the model to a file."""
    save_dir = get_artifacts_dir(Config.MODEL_DIR)
    model.save(os.path.join(save_dir, f"model_{Config.TIMESTAMP}.keras"))


def checkpoint_path():
    """Returns the file path for saving model checkpoints."""
    save_dir = get_artifacts_dir(Config.CHECKPOINT_DIR)

    # Use a custom file path with placeholders for epoch and validation loss
    return os.path.join(
        save_dir,
        f"model_checkpoint_{Config.TIMESTAMP}_epoch-{{epoch:02d}}_val-loss-{{val_loss:.4f}}.keras",
    )


def plot_history(history):
    """Plots the training history and saves the plot to a file."""
    save_dir = get_artifacts_dir(Config.PLOT_DIR)
    save_file_path = os.path.join(save_dir, f"history_{Config.TIMESTAMP}.png")

    # Create a new figure for the plot
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

    # Add title and labels to the plot
    plt.title("Training, Validation, and Test Loss/Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_file_path)

    # Close the plot to free up memory
    plt.close()


def save_history(history):
    """Saves the training history as a JSON file."""
    save_dir = get_artifacts_dir(Config.HISTORY_DIR)
    save_file_path = os.path.join(save_dir, f"history_{Config.TIMESTAMP}.json")

    # Save the history dictionary to a JSON file
    with open(save_file_path, "w") as file:
        json.dump(history.history, file)
