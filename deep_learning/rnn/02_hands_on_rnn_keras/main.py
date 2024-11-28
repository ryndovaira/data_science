"""
Main module to execute the end-to-end pipeline for data loading,
model tuning, training, and evaluation.
"""

import pandas as pd
from tuner import tune_hyperparameters, retrain_with_best_hps
from eda_and_preprocessing import load_dataset_compute_length_buckets, load_and_preprocess_data
from logger import setup_logger
from plotter import plot_all_results
from saver import save_all_results
from config import Config

# Setup logger
logger = setup_logger()


def main():
    """Main function to automate experiments across length buckets."""

    logger.info("Starting pipeline.")
    logger.info("Loading dataset and computing length buckets.")
    length_buckets = load_dataset_compute_length_buckets()
    logger.info("Dataset loaded and length buckets computed.")

    results = []

    architectures = ["VanillaRNN", "LSTM", "GRU"]
    logger.info(f"Testing architectures: {architectures}")
    for arch in architectures:
        Config.ARCHITECTURE = arch
        logger.info(f"Testing architecture: {arch}")
        for min_len, max_len in length_buckets:
            logger.info(f"Starting experiment for length bucket: {min_len}-{max_len}.")
            Config.MIN_LEN = min_len
            Config.MAX_LEN = max_len

            try:
                (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
                    Config.MIN_LEN, Config.MAX_LEN, Config.MAX_FEATURES
                )

                best_hps = tune_hyperparameters(x_train, y_train, x_val, y_val)

                train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = (
                    retrain_with_best_hps(best_hps, x_train, y_train, x_val, y_val, x_test, y_test)
                )

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

    logger.info("All experiments completed.")

    logger.info("Saving results.")
    results_df = pd.DataFrame(results)
    logger.info(f"Results: {results_df}")
    save_all_results(results_df)
    logger.info("Results saved.")

    logger.info("Plotting results.")
    plot_all_results(results_df)
    logger.info("Results plotted.")

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
