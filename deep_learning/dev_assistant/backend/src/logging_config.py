import logging
from pathlib import Path
from datetime import datetime


class LogConfig:
    """
    Configuration for logging, including log directory, file naming, and format.
    """

    # Define the log directory outside the src directory, at the project root
    LOG_DIR = (
        Path(__file__).resolve().parents[2] / "logs"
    )  # Adjust path relative to the project root
    LOG_DIR.mkdir(exist_ok=True)  # Ensure the directory exists

    # Define a consistent timestamp for the log file
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the full path for the log file
    @classmethod
    def get_log_file_path(cls):
        """
        Returns the full path to the log file for the current application run.
        """
        return cls.LOG_DIR / f"log_{cls.TIMESTAMP}.log"


def setup_logger(name=None):
    """
    Sets up a logger that writes logs to a timestamped file and the console.

    Args:
        name (str): Name of the logger. If None, the root logger is used.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get the log file path from LogConfig
    log_file_path = LogConfig.get_log_file_path()

    # Define the log format
    log_format = "%(asctime)s - [%(levelname)s] %(name)s: %(message)s"

    # Create and configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(log_format)
    stream_handler.setFormatter(stream_formatter)

    # Avoid adding multiple handlers if the logger is reused
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
