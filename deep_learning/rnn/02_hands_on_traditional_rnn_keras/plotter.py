import os
import logging

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from utils import get_artifacts_dir

# Get the existing logger configured in main.py
logger = logging.getLogger()


# Define the Dracula theme
dracula_template = {
    "layout": {
        "paper_bgcolor": "#282a36",  # Background color
        "plot_bgcolor": "#282a36",  # Plot area background
        "font": {
            "color": "#f8f8f2",  # Text color
            "size": 12,
        },
        "colorway": [
            "#ff5555",  # Red
            "#50fa7b",  # Green
            "#bd93f9",  # Purple
            "#f1fa8c",  # Yellow
            "#6272a4",  # Blue
        ],
        "xaxis": {
            "gridcolor": "#44475a",  # Grid lines
            "zerolinecolor": "#44475a",  # Zero line
            "tickcolor": "#f8f8f2",
        },
        "yaxis": {
            "gridcolor": "#44475a",
            "zerolinecolor": "#44475a",
            "tickcolor": "#f8f8f2",
        },
        "legend": {
            "bgcolor": "#282a36",  # Legend background
            "bordercolor": "#44475a",  # Legend border
            "font": {"color": "#f8f8f2"},
        },
        "coloraxis": {
            "colorbar": {
                "outlinecolor": "#f8f8f2",  # Colorbar border
                "tickcolor": "#f8f8f2",
            }
        },
    }
}

# Register the template and set it as default
pio.templates["dracula"] = dracula_template
pio.templates.default = "dracula"


def plot_history(history):
    """Plots the training history and saves the plot to a file using Plotly."""
    save_dir = get_artifacts_dir(Config.PLOT_DIR)
    save_file_path = os.path.join(
        save_dir, f"history_{Config.min_max_len()}_{Config.TIMESTAMP}.html"
    )

    # Create the epoch list
    epochs = list(range(1, len(history.history["loss"]) + 1))

    # Create the figure
    fig = go.Figure()

    # Add training loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["loss"],
            mode="markers+lines",
            name="Training Loss",
            marker=dict(color="blue"),
        )
    )
    # Add validation loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["val_loss"],
            mode="markers+lines",
            name="Validation Loss",
            marker=dict(color="orange"),
        )
    )
    # Add training accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["accuracy"],
            mode="markers+lines",
            name="Training Accuracy",
            marker=dict(color="green"),
        )
    )
    # Add validation accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["val_accuracy"],
            mode="markers+lines",
            name="Validation Accuracy",
            marker=dict(color="red"),
        )
    )

    # Add vertical and horizontal traces for each epoch
    for epoch, train_loss, val_loss, train_acc, val_acc in zip(
        epochs,
        history.history["loss"],
        history.history["val_loss"],
        history.history["accuracy"],
        history.history["val_accuracy"],
    ):
        # Add numeric annotations near each point
        fig.add_trace(
            go.Scatter(
                x=[epoch],
                y=[train_loss],
                mode="text",
                text=[f"{train_loss:.3f}"],
                textposition="top center",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[epoch],
                y=[val_loss],
                mode="text",
                text=[f"{val_loss:.3f}"],
                textposition="top center",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[epoch],
                y=[train_acc],
                mode="text",
                text=[f"{train_acc:.3f}"],
                textposition="bottom center",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[epoch],
                y=[val_acc],
                mode="text",
                text=[f"{val_acc:.3f}"],
                textposition="bottom center",
                showlegend=False,
            )
        )

    # Add layout details
    fig.update_layout(
        title="Training and Validation Metrics Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Metric Value",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    # Save a plot as an HTML file
    fig.write_html(save_file_path)

    logger.info(f"History plot saved to {save_file_path}.")


def plot_all_results(results_df):
    """Generate an interactive plot to compare accuracy across length buckets using Plotly."""
    save_dir = get_artifacts_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, f"accuracy_vs_length_{Config.TIMESTAMP}.html")

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
        showlegend=True,
    )

    # Save a plot as an HTML file
    fig.write_html(save_file_path)

    logger.info(f"Comparison plot saved to {save_file_path}.")
