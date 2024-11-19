class Config:
    MAX_FEATURES = 10000  # Maximum number of words to keep in the vocabulary
    MAX_LEN = 500  # Maximum length of input sequences (long sequences)
    SHORT_MAX_LEN = 50  # Maximum length for short input sequences
    EPOCHS = 2  # Number of training epochs
    BATCH_SIZE = 2  # Number of samples per training batch
    EMBEDDING_DIM = 32  # Dimensionality of the embedding layer
    RNN_UNITS = 16  # Number of units in the RNN layer
    DEV_MODE = True  # Toggle for development mode (using fewer samples)
    DEV_SAMPLES = 100  # Number of samples to use when in development mode
    MODE = "mixed"  # Mode for data preprocessing: 'short', 'long', or 'mixed'
    LOG_DIR = "logs"  # Directory for log files
    ARTIFACTS_DIR = "artifacts"  # Directory for saving model artifacts
    PLOT_DIR = "plots"  # Directory for saving plots
