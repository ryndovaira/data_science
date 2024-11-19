import logging
import os
from datetime import datetime


def setup_logger(name):
    """Sets up a logger that writes to a specified log file with a unique timestamp."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add timestamp to the log file name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_file_name = f"{name}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Create and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers if the logger is reused
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def save_artifacts(history, model, output_dir):
    """Saves training history and model artifacts."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'final_model.h5'))
    # Optionally, save history as a plot or JSON
