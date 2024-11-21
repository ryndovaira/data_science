import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from data_preprocessing import load_and_preprocess_data
from config import Config
from utils import setup_logger, save_model, checkpoint_path, plot_history, save_history, get_artifacts_dir
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

# Setup logger
logger = setup_logger()


def model_builder(hp):
    """Builds the model for hyperparameter tuning."""
    embedding_dim = hp.Choice("embedding_dim", values=[32, 64, 128])
    rnn_units = hp.Int("rnn_units", min_value=16, max_value=128, step=16)

    model = Sequential(
        [
            Embedding(
                input_dim=Config.MAX_FEATURES, output_dim=embedding_dim
            ),
            SimpleRNN(rnn_units),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def tune_hyperparameters():
    """Tunes hyperparameters using Keras Tuner and generates trial plots."""
    logger.info("Starting hyperparameter tuning.")

    # Pass dummy max features for initial data loading (updated dynamically during tuning)
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess_data(
        Config.MIN_LEN, Config.MAX_LEN, Config.MAX_FEATURES
    )

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=Config.TUNER_MAX_EPOCHS,
        factor=3,
        directory=get_artifacts_dir(Config.TUNER_DIR),
        project_name="trials",
    )
    try:
        tuner.search(
            x_train,
            y_train,
            epochs=Config.TUNER_MAX_EPOCHS,
            validation_data=(x_val, y_val),
            batch_size=Config.BATCH_SIZE,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
        )
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"Best hyperparameters found: {best_hps.values}")

    # Save and plot tuning results
    save_tuner_results(tuner)
    plot_tuner_trials(tuner)

    return best_hps


def save_tuner_results(tuner, num_trials=10):
    """Saves tuner trial results as a CSV file."""
    results = []
    trials = tuner.oracle.get_best_trials(num_trials=num_trials)  # Specify an integer for num_trials
    for trial in trials:
        trial_data = trial.hyperparameters.values
        trial_data["val_accuracy"] = trial.score
        results.append(trial_data)

    results_df = pd.DataFrame(results)
    save_dir = get_artifacts_dir(Config.TUNER_DIR)
    save_file_path = os.path.join(save_dir, f"tuner_results_{Config.TIMESTAMP}.csv")
    results_df.to_csv(save_file_path, index=False)
    logger.info(f"Tuner results saved to {save_file_path}")


def plot_tuner_trials(tuner):
    """
    Generates a heatmap to compare validation accuracy across trials for all hyperparameters,
    with the best trial highlighted.
    """
    trials = list(tuner.oracle.trials.values())  # Retrieve all trials
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]  # Get the best trial

    save_dir = get_artifacts_dir(Config.PLOT_DIR)
    save_file_path = os.path.join(save_dir, f"trial_comparison_heatmap_{Config.TIMESTAMP}.png")

    # Consolidate all trial data into a DataFrame for easier plotting
    trial_data = []
    for trial in trials:
        trial_info = trial.hyperparameters.values.copy()
        trial_info["val_accuracy"] = trial.score
        trial_info["trial_id"] = trial.trial_id
        trial_data.append(trial_info)

    df_trials = pd.DataFrame(trial_data)

    # Add a column to indicate the best trial
    df_trials["is_best"] = df_trials["trial_id"] == best_trial.trial_id

    # Convert categorical hyperparameters to string for proper heatmap indexing
    df_numeric = df_trials.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == object:
            df_numeric[col] = df_numeric[col].astype(str)

    # Select only numeric columns for the heatmap
    heatmap_data = df_numeric.set_index("trial_id").select_dtypes(include=[np.number])

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Validation Accuracy"},
        linewidths=0.5,
    )

    # Highlight the best trial
    best_trial_id = best_trial.trial_id
    if best_trial_id in heatmap_data.index:
        row_idx = list(heatmap_data.index).index(best_trial_id)
        ax.add_patch(plt.Rectangle((0, row_idx), heatmap_data.shape[1], 1, fill=False, edgecolor='red', lw=3))

    plt.title("Validation Accuracy Across Trials (Best Trial Highlighted)")
    plt.xlabel("Hyperparameters & Validation Accuracy")
    plt.ylabel("Trial ID")
    plt.tight_layout()
    plt.savefig(save_file_path)
    plt.close()
    logger.info(f"Saved heatmap of all trials to {save_file_path}")


def retrain_with_best_hps(best_hps):
    """Retrains the model using the best hyperparameters."""
    logger.info("Retraining model with best hyperparameters.")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(
        Config.MIN_LEN, Config.MAX_LEN, Config.MAX_FEATURES
    )

    model = model_builder(best_hps)
    history = model.fit(
        x_train,
        y_train,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path(), monitor="val_loss", save_best_only=True),
        ],
    )

    save_model(model)
    plot_history(history)
    save_history(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
