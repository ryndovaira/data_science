import json
import logging
import os
import pandas as pd
import plotly.graph_objects as go

from config import Config

# Get the existing logger configured in main.py
logger = logging.getLogger()


def get_artifacts_dir(base_dir: str, *sub_dirs: str) -> str:
    """Returns the directory path for saving artifacts, ensuring it exists."""
    dir_path = os.path.join(os.getcwd(), Config.ARTIFACTS_DIR, base_dir, Config.name(), *sub_dirs)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def setup_logger():
    """Sets up a logger that writes to a specified log file with a unique timestamp."""

    # Define the log directory path
    save_dir = get_artifacts_dir(Config.LOG_DIR)

    # Define the log file path with a timestamp
    log_file_path = os.path.join(save_dir, f"log_{Config.TIMESTAMP}.txt")

    # Create and configure the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Avoid adding multiple handlers if the logger is reused
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def save_model(model: "tf.keras.Model"):
    """Saves the model to a file."""
    save_dir = get_artifacts_dir(Config.MODEL_DIR)
    model.save(os.path.join(save_dir, f"model_{Config.TIMESTAMP}.keras"))
    logger.info(f"Model saved to {save_dir}.")


def checkpoint_path():
    """Returns the file path for saving model checkpoints."""
    save_dir = get_artifacts_dir(Config.CHECKPOINT_DIR)

    # Use a custom file path with placeholders for epoch and validation loss
    return os.path.join(
        save_dir,
        f"model_checkpoint_{Config.TIMESTAMP}_epoch-{{epoch:02d}}_val-loss-{{val_loss:.4f}}.keras",
    )


def plot_history(history):
    """Plots the training history and saves the plot to a file using Plotly."""
    save_dir = get_artifacts_dir(Config.PLOT_DIR)
    save_file_path = os.path.join(save_dir, f"history_{Config.TIMESTAMP}.html")

    # Create traces for loss
    loss_trace = go.Scatter(y=history.history["loss"], mode="lines", name="Training Loss")
    val_loss_trace = go.Scatter(y=history.history["val_loss"], mode="lines", name="Validation Loss")

    # Create traces for accuracy
    accuracy_trace = go.Scatter(
        y=history.history["accuracy"], mode="lines", name="Training Accuracy"
    )
    val_accuracy_trace = go.Scatter(
        y=history.history["val_accuracy"], mode="lines", name="Validation Accuracy"
    )

    # Combine traces into a figure
    fig = go.Figure(data=[loss_trace, val_loss_trace, accuracy_trace, val_accuracy_trace])

    # Add layout details
    fig.update_layout(
        title="Training and Validation Metrics Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    # Save a plot as an HTML file
    fig.write_html(save_file_path)

    logger.info(f"History plot saved to {save_file_path}.")


def save_history(history):
    """Saves the training history as a JSON file."""
    save_dir = get_artifacts_dir(Config.HISTORY_DIR)
    save_file_path = os.path.join(save_dir, f"history_{Config.TIMESTAMP}.json")

    # Save the history dictionary to a JSON file
    with open(save_file_path, "w") as file:
        json.dump(history.history, file)

    logger.info(f"History saved to {save_file_path}.")


def plot_all_results(results_df):
    """Generate an interactive plot to compare accuracy across length buckets using Plotly."""
    save_dir = get_artifacts_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, "accuracy_vs_length.html")

    # Create a scatter plot for accuracy vs. sequence length
    fig = go.Figure()

    # Add the main scatter plot
    fig.add_trace(
        go.Scatter(
            x=results_df["max_len"],
            y=results_df["test_accuracy"],
            mode="markers+lines+text",
            marker=dict(size=10, color="blue"),
            name="Test Accuracy",
            text=results_df["test_accuracy"].round(3),  # Display accuracy rounded to 3 decimals
            textposition="top center",  # Position text above markers
        )
    )

    # Add vertical and horizontal reference lines (traces to X and Y axes) and value annotations
    for i, row in results_df.iterrows():
        # Vertical line to X-axis
        fig.add_shape(
            type="line",
            x0=row["max_len"],
            x1=row["max_len"],
            y0=0,
            y1=row["test_accuracy"],
            line=dict(color="gray", dash="dot"),
        )
        # Horizontal line to Y-axis
        fig.add_shape(
            type="line",
            x0=0,
            x1=row["max_len"],
            y0=row["test_accuracy"],
            y1=row["test_accuracy"],
            line=dict(color="gray", dash="dot"),
        )
        # Add X-axis value annotation
        fig.add_trace(
            go.Scatter(
                x=[row["max_len"]],
                y=[0],
                mode="text",
                text=[str(row["max_len"])],
                textposition="bottom center",
                showlegend=False,
            )
        )
        # Add Y-axis value annotation
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[row["test_accuracy"]],
                mode="text",
                text=[f"{row['test_accuracy']:.3f}"],
                textposition="middle right",
                showlegend=False,
            )
        )

    # Add layout details
    fig.update_layout(
        title="Test Accuracy vs. Sequence Length",
        xaxis_title="Max Sequence Length",
        yaxis_title="Test Accuracy",
        template="plotly_dark",
        showlegend=True,
    )

    # Save a plot as an HTML file
    fig.write_html(save_file_path)

    logger.info(f"Comparison plot saved to {save_file_path}.")


def save_all_results(df):
    """Save all results to a CSV file."""
    save_dir = get_artifacts_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, f"length_bucket_results_{Config.TIMESTAMP}.csv")
    df.to_csv(save_file_path, index=False)

    logger.info(f"Results saved to {save_file_path}.")


def save_tuner_results(tuner, num_trials=10):
    """Saves tuner trial results as a CSV file."""
    results = []
    trials = tuner.oracle.get_best_trials(
        num_trials=num_trials
    )  # Specify an integer for num_trials
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
    """Generates an interactive heatmap to compare validation accuracy across trials using Plotly."""
    trials = list(tuner.oracle.trials.values())  # Retrieve all trials
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]  # Get the best trial

    save_dir = get_artifacts_dir(Config.PLOT_DIR)
    save_file_path = os.path.join(save_dir, f"trial_comparison_heatmap_{Config.TIMESTAMP}.html")

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

    # Generate a heatmap using Plotly
    heatmap_data = df_trials.pivot(index="trial_id", values="val_accuracy", columns="rnn_units")

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Viridis",
            colorbar=dict(title="Validation Accuracy"),
        )
    )

    # Highlight the best trial and add text annotations for values
    fig.add_trace(
        go.Scatter(
            x=[best_trial.hyperparameters.get("rnn_units")],
            y=[best_trial.trial_id],
            mode="markers+text",
            marker=dict(size=12, color="red"),
            name="Best Trial",
            text=[f"Best: {best_trial.score:.3f}"],  # Show the best score
            textposition="bottom right",
        )
    )

    # Add layout details
    fig.update_layout(
        title="Validation Accuracy Across Trials",
        xaxis_title="RNN Units",
        yaxis_title="Trial ID",
        template="plotly_dark",
    )

    # Save the heatmap as an HTML file
    fig.write_html(save_file_path)

    logger.info(f"Saved heatmap of all trials to {save_file_path}.")
