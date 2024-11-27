import logging
import os

from config import Config
from utils import get_artifacts_dir


def setup_logger():
    """Sets up a logger that writes to a specified log file with a unique timestamp."""

    # Define the log directory path
    save_dir = os.path.join(os.getcwd(), Config.ARTIFACTS_DIR, Config.LOG_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # Define the log file path with a timestamp
    log_file_path = os.path.join(save_dir, f"log_{Config.TIMESTAMP}.txt")

    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Create and configure the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(log_format)
    stream_handler.setFormatter(stream_formatter)

    # Avoid adding multiple handlers if the logger is reused
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(stream_handler)

    return logger
