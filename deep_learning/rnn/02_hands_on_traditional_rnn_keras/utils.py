import logging
import os


def setup_logger(log_path):
    """Sets up a logger that writes to a specified log file."""
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def save_artifacts(history, model, output_dir):
    """Saves training history and model artifacts."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'final_model.h5'))
    # Optionally, save history as a plot or JSON
