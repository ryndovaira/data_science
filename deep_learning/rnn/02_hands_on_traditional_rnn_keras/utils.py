import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from config import Config


def get_timestamp() -> str:
    """Returns the current timestamp as a string."""
    return datetime.now().strftime("%y%m%d_%H%M%S")


def setup_logger():
    """Sets up a logger that writes to a specified log file with a unique timestamp."""
    # Define the log directory path
    log_dir = os.path.join(os.getcwd(), Config.LOG_DIR, Config.MODE)
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define the log file path with a timestamp
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
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the model filename with a timestamp
    model_filename = f"model_{get_timestamp()}.keras"
    # Save the model to the specified directory
    model.save(os.path.join(output_dir, Config.MODE, model_filename))


def checkpoint_path():
    """Generates a unique checkpoint path with a timestamp."""
    # Define the checkpoint directory path
    checkpoint_dir = os.path.join(Config.ARTIFACTS_DIR, Config.MODE)
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint file path with placeholders for epoch and validation loss
    return os.path.join(
        checkpoint_dir,
        f"model_checkpoint_{get_timestamp()}_epoch-{{epoch:02d}}_val-loss-{{val_loss:.4f}}.keras",
    )


def plot_history(history):
    """Plots and saves the training history as an image file."""
    # Define the save directory path
    save_dir = os.path.join(os.getcwd(), Config.PLOT_DIR, Config.MODE)
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Define the save file path with a timestamp
    save_file_path = os.path.join(save_dir, f"plot_{get_timestamp()}.png")

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

    # Save the plot to the specified file
    plt.savefig(save_file_path)
    plt.close()
