from tuner import tune_hyperparameters, retrain_with_best_hps
from utils import setup_logger

# Setup logger
logger = setup_logger()


def main():
    """Main function to run hyperparameter tuning and retraining."""
    logger.info("Starting hyperparameter tuning.")
    best_hps = tune_hyperparameters()  # Tune hyperparameters
    logger.info("Hyperparameter tuning completed.")

    logger.info("Retraining the model with the best hyperparameters.")
    retrain_with_best_hps(best_hps)  # Retrain with best hyperparameters


if __name__ == "__main__":
    main()
