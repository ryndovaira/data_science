"""
Main module to execute the end-to-end pipeline for data loading,
model tuning, training, and evaluation.
"""

from tuner import tune_hyperparameters, retrain_with_best_hps
from utils import setup_logger

# Setup logger
logger = setup_logger()


def main():
    """Main function to run hyperparameter tuning and retraining."""
    try:
        logger.info("Starting hyperparameter tuning.")
        best_hps = tune_hyperparameters()  # Tune hyperparameters
        logger.info("Hyperparameter tuning completed.")

        logger.info("Retraining the model with the best hyperparameters.")
        retrain_with_best_hps(best_hps)  # Retrain with the best hyperparameters
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
