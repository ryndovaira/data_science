import json
import os
import logging

import pandas as pd
from keras_tuner.src.backend.io import tf

from src.config import Config
from utils import get_artifacts_arch_dir, get_artifacts_dir

# Get the existing logger configured in main.py
logger = logging.getLogger()


def save_model(model: tf.keras.Model):
    """Saves the model to a file."""
    logger.info("Saving model.")
    save_dir = get_artifacts_arch_dir(Config.MODEL_DIR)
    model.save(os.path.join(save_dir, f"model_{Config.min_max_len()}_{Config.TIMESTAMP}.keras"))
    logger.info(f"Model saved to {save_dir}.")


def save_history(history):
    """Saves the training history as a JSON file."""
    logger.info("Saving history.")
    save_dir = get_artifacts_arch_dir(Config.HISTORY_DIR)
    save_file_path = os.path.join(
        save_dir, f"history_{Config.min_max_len()}_{Config.TIMESTAMP}.json"
    )

    # Save the history dictionary to a JSON file
    with open(save_file_path, "w") as file:
        json.dump(history.history, file)

    logger.info(f"History saved to {save_file_path}.")


def save_all_results(df):
    """Save all results to a CSV file."""
    logger.info("Saving all results.")
    save_dir = get_artifacts_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, f"length_bucket_results_{Config.TIMESTAMP}.csv")
    df.to_csv(save_file_path, index=False)

    logger.info(f"Results saved to {save_file_path}.")


def save_tuner_results(tuner, num_trials=10):
    """Saves tuner trial results as a CSV file."""
    logger.info("Saving tuner results.")
    results = []
    trials = tuner.oracle.get_best_trials(num_trials=num_trials)
    for trial in trials:
        trial_data = trial.hyperparameters.values
        trial_data["val_accuracy"] = trial.score
        results.append(trial_data)

    results_df = pd.DataFrame(results)
    save_dir = get_artifacts_arch_dir(Config.TUNER_DIR)
    save_file_path = os.path.join(
        save_dir, f"tuner_results_{Config.min_max_len()}_{Config.TIMESTAMP}.csv"
    )
    results_df.to_csv(save_file_path, index=False)

    logger.info(f"Tuner results saved to {save_file_path}")
