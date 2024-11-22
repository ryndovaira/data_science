from datetime import datetime


class Config:
    """Configuration for the project"""
    MAX_FEATURES = 5000  # Maximum number of words to keep in the vocabulary

    # Sequence length buckets
    # 0-130, 130-175, 175-285, 285-590, 590-1000
    MIN_LEN = 130
    MAX_LEN = 175

    EPOCHS = 50  # Number of training epochs
    TUNER_MAX_EPOCHS = 10  # Number of epochs for hyperparameter tuning
    BATCH_SIZE = 32  # Number of samples per training batch
    DEV_MODE = False  # Toggle for development mode (using fewer samples)
    DEV_SAMPLES = 100  # Number of samples to use when in development mode

    RANDOM_SEED = 42  # Random seed for reproducibility

    # Hyperband parameters
    HYPERBAND_FACTOR = 3
    HYPERBAND_ITERATIONS = 4
    HYPERBAND_PROJ_NAME = "trials"

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
        return f"{cls.MIN_LEN}_{cls.MAX_LEN}_{cls.HYPERBAND_FACTOR}_{cls.HYPERBAND_ITERATIONS}"
