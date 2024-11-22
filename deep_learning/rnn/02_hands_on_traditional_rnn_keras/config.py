from datetime import datetime


class Config:
    """Configuration for the project"""
    MAX_FEATURES = 5000  # Maximum number of words to keep in the vocabulary
    MIN_LEN = 0  # Minimum length of input sequences
    MAX_LEN = 15  # Maximum length of input sequences
    EPOCHS = 50  # Number of training epochs
    TUNER_MAX_EPOCHS = 10  # Number of epochs for hyperparameter tuning
    BATCH_SIZE = 2  # Number of samples per training batch
    DEV_MODE = False  # Toggle for development mode (using fewer samples)
    DEV_SAMPLES = 100  # Number of samples to use when in development mode

    RANDOM_SEED = 42  # Random seed for reproducibility

    HYPERBAND_FACTOR = 3  # Factor for Hyperband tuning
    HYPERBAND_ITERATIONS = 1  # Number of Hyperband iterations
    HYPERBAND_PROJ_NAME = "trials"  # Project name for Hyperband tuner

    # Artifact directories
    LOG_DIR = "logs"
    ARTIFACTS_DIR = "artifacts"
    MODEL_DIR = "model"
    PLOT_DIR = "plots"
    HISTORY_DIR = "history"
    TUNER_DIR = "tuner_results"
    CHECKPOINT_DIR = "checkpoints"

    TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")  # Timestamp for filenames

    @classmethod
    def name(cls) -> str:
        return f"{cls.MIN_LEN}_{cls.MAX_LEN}"
