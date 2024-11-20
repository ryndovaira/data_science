from datetime import datetime


class Config:
    MAX_FEATURES = 10000  # Maximum number of words to keep in the vocabulary
    MIN_LEN = 0  # Minimum length of input sequences
    MAX_LEN = 15  # Maximum length of input sequences
    EPOCHS = 2  # Number of training epochs
    BATCH_SIZE = 2  # Number of samples per training batch
    EMBEDDING_DIM = 32  # Dimensionality of the embedding layer
    RNN_UNITS = 16  # Number of units in the RNN layer
    DEV_MODE = True  # Toggle for development mode (using fewer samples)
    DEV_SAMPLES = 100  # Number of samples to use when in development mode
    LOG_DIR = "logs"  # Directory for log files
    ARTIFACTS_DIR = "artifacts"  # Directory for saving model artifacts
    PLOT_DIR = "plot"  # Directory for saving plots
    MODEL_DIR = "model"  # Directory for saving model files
    HISTORY_DIR = "history"  # Directory for saving training history
    RANDOM_SEED = 42  # Random seed for reproducibility
    TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")  # Timestamp for unique names

    @classmethod
    def name(cls) -> str:
        return f"{cls.MIN_LEN}_{cls.MAX_LEN}_{cls.EPOCHS}_{cls.BATCH_SIZE}_{cls.EMBEDDING_DIM}_{cls.RNN_UNITS}_{cls.MAX_FEATURES}"
