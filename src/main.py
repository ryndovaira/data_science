"""
Main module to execute the end-to-end pipeline for data loading,
model tuning, training, and evaluation.
"""

import time
import pandas as pd
from tuner import tune_hyperparameters, retrain_with_best_hps
from src.data_preprocessing import (
    compute_length_buckets,
    preprocess_data,
    load_dataset,
)
from logger import setup_logger
from plotter import plot_all_results
from saver import save_all_results
from src.config import Config

# Setup logger
logger = setup_logger()


def main():
    """Main function to automate experiments across length buckets."""

    try:
        logger.info("Starting pipeline.")
        start_time = time.time()

        x_train, y_train, x_test, y_test = load_dataset()

        length_buckets = compute_length_buckets(x_train, x_test)

        preprocessed_data = dict()
        for min_len, max_len in length_buckets:
            preprocessed_data[(min_len, max_len)] = preprocess_data(
                x_train,
                y_train,
                x_test,
                y_test,
                min_len,
                max_len,
                Config.MAX_FEATURES,
            )
    except Exception as e:
        logger.error(f"Data loading and preprocessing failed: {e}")
        return

    results = []

    architectures = ["VanillaRNN", "LSTM", "GRU"]
    for arch in architectures:
        Config.ARCHITECTURE = arch
        for min_len, max_len in length_buckets:
            logger.info(f"Starting experiment for length bucket {min_len}-{max_len} with {arch}.")
            Config.MIN_LEN = min_len
            Config.MAX_LEN = max_len

            experiment_start_time = time.time()

            try:
                (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocessed_data[
                    (min_len, max_len)
                ]

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

                logger.info(
                    f"Experiment completed for {arch} with length bucket {min_len}-{max_len}. "
                    f"Time taken: {time.time() - experiment_start_time:.2f} seconds."
                )

            except Exception as e:
                logger.error(f"Experiment failed for {min_len}-{max_len} with {arch}: {e}")

    total_time = time.time() - start_time
    logger.info(f"All experiments completed. Total time: {total_time:.2f} seconds.")

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
