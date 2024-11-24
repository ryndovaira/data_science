"""
Main module to execute the end-to-end pipeline for data loading,
model tuning, training, and evaluation.
"""

import pandas as pd
from tuner import tune_hyperparameters, retrain_with_best_hps
from logger import setup_logger
from plotter import plot_all_results
from saver import save_all_results
from config import Config

# Setup logger
logger = setup_logger()

# Define sequence length buckets
LENGTH_BUCKETS = [
    (0, 130),
    (130, 175),
    (175, 285),
    (285, 590),
    (590, 1000),
]


def main():
    """Main function to automate experiments across length buckets."""
    results = []  # Store results for comparison

    for min_len, max_len in LENGTH_BUCKETS:
        logger.info(f"Starting experiment for length bucket: {min_len}-{max_len}.")

        # Update Config dynamically
        Config.MIN_LEN = min_len
        Config.MAX_LEN = max_len

        try:
            # Hyperparameter tuning
            logger.info("Starting hyperparameter tuning.")
            best_hps = tune_hyperparameters()
            logger.info(f"Hyperparameter tuning completed for {min_len}-{max_len}.")

            # Retrain with best hyperparameters
            logger.info("Retraining model with best hyperparameters.")
            test_loss, test_accuracy = retrain_with_best_hps(best_hps)

            # Record results
            results.append(
                {
                    "min_len": min_len,
                    "max_len": max_len,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "best_hyperparameters": best_hps.values,
                }
            )

        except Exception as e:
            logger.error(
                f"Experiment failed for length bucket {min_len}-{max_len}: {e}", exc_info=True
            )

    # Save and plot results
    results_df = pd.DataFrame(results)
    save_all_results(results_df)
    plot_all_results(results_df)


if __name__ == "__main__":
    main()
