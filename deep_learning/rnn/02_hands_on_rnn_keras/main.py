"""
Main module to execute the end-to-end pipeline for data loading,
model tuning, training, and evaluation.
"""

import pandas as pd
from tuner import tune_hyperparameters, retrain_with_best_hps
from eda import load_dataset_compute_length_buckets
from logger import setup_logger
from plotter import plot_all_results
from saver import save_all_results
from config import Config

# Setup logger
logger = setup_logger()


def main():
    """Main function to automate experiments across length buckets."""

    length_buckets = load_dataset_compute_length_buckets()

    results = []

    # Iterate through different architectures
    architectures = ["VanillaRNN", "LSTM", "GRU"]
    for arch in architectures:
        Config.ARCHITECTURE = arch
        logger.info(f"Testing architecture: {arch}")

        for min_len, max_len in length_buckets:
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
                logger.info("Retraining model with the best hyperparameters.")
                test_loss, test_accuracy, val_loss, val_accuracy, train_loss, train_accuracy = (
                    retrain_with_best_hps(best_hps)
                )

                # Record results
                results.append(
                    {
                        "architecture": Config.ARCHITECTURE,
                        "min_len": min_len,
                        "max_len": max_len,
                        "test_loss": test_loss,
                        "test_accuracy": test_accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
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
