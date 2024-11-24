import logging
import os

from config import Config
from utils import get_artifacts_dir


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
