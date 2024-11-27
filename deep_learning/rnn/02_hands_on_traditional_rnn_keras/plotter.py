import os
import logging

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from utils import get_artifacts_arch_dir, get_artifacts_dir

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
    """Plots the training history as two subplots: one for loss and one for accuracy, with numeric annotations."""
    save_dir = get_artifacts_arch_dir(Config.PLOT_DIR)
    save_file_path = os.path.join(
        save_dir, f"history_{Config.min_max_len()}_{Config.TIMESTAMP}.html"
    )

    epochs = list(range(1, len(history.history["loss"]) + 1))

    training_color = "#50fa7b"  # Green for training metrics
    validation_color = "#ff5555"  # Red for validation metrics

    # Create subplots with proper vertical spacing
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,  # Separate X-axes to allow tick labels on both subplots
        vertical_spacing=0.1,  # Spacing between subplots
        subplot_titles=("Loss Over Epochs", "Accuracy Over Epochs"),
    )

    # Add loss traces with numeric annotations
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["loss"],
            mode="markers+lines+text",
            name="Training Loss",
            marker=dict(color=training_color),
            text=[f"{val:.3f}" for val in history.history["loss"]],
            textposition="top center",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["val_loss"],
            mode="markers+lines+text",
            name="Validation Loss",
            marker=dict(color=validation_color),
            text=[f"{val:.3f}" for val in history.history["val_loss"]],
            textposition="top center",
        ),
        row=1,
        col=1,
    )

    # Add accuracy traces with numeric annotations
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["accuracy"],
            mode="markers+lines+text",
            name="Training Accuracy",
            marker=dict(color=training_color),
            text=[f"{val:.3f}" for val in history.history["accuracy"]],
            textposition="top center",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.history["val_accuracy"],
            mode="markers+lines+text",
            name="Validation Accuracy",
            marker=dict(color=validation_color),
            text=[f"{val:.3f}" for val in history.history["val_accuracy"]],
            textposition="top center",
        ),
        row=2,
        col=1,
    )

    # Update layout for title, legend, and margins
    fig.update_layout(
        height=800,  # Increase height for better spacing
        title=dict(
            text="Training and Validation Metrics Over Epochs",
            y=0.98,  # Place title near the top
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.05,  # Position directly below the title
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            t=120,  # Top margin for title and legend
            b=50,  # Bottom margin
            l=60,  # Left margin
            r=60,  # Right margin
        ),
    )

    # Add titles and tick labels to X and Y axes
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_xaxes(
        tickmode="linear",
        row=1,
        col=1,
    )  # Add numeric ticks for the top plot
    fig.update_xaxes(
        title_text="Epochs", tickmode="linear", row=2, col=1
    )  # Numeric ticks for the bottom plot

    # Save the plot
    fig.write_html(save_file_path)
    logger.info(f"History plot saved to {save_file_path}.")


def plot_all_results_old(results_df):
    """Generate an interactive plot to compare accuracy across length buckets using Plotly."""
    save_dir = get_artifacts_arch_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, f"accuracy_vs_length_{Config.TIMESTAMP}.html")

    fig = go.Figure()

    # Add scatter plot for accuracy vs. sequence length
    fig.add_trace(
        go.Scatter(
            x=results_df["max_len"],
            y=results_df["test_accuracy"],
            mode="markers+lines+text",
            text=results_df["test_accuracy"].round(3),
            name="Test Accuracy",
        )
    )

    # Add annotations only for outliers or best points
    threshold = results_df["test_accuracy"].mean() + results_df["test_accuracy"].std()
    for _, row in results_df.iterrows():
        if row["test_accuracy"] > threshold:
            fig.add_trace(
                go.Scatter(
                    x=[row["max_len"]],
                    y=[row["test_accuracy"]],
                    mode="text",
                    text=[f"{row['test_accuracy']:.3f}"],
                    textposition="top center",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title="Test Accuracy vs. Sequence Length",
        xaxis_title="Max Sequence Length",
        yaxis_title="Test Accuracy",
        legend=dict(orientation="v", xanchor="right", x=1.05, yanchor="top", y=1.5),
    )

    # Save a plot as an HTML file
    fig.write_html(save_file_path)

    logger.info(f"Comparison plot saved to {save_file_path}.")


def plot_all_results(results_df):
    """
    Generate a single combined plot for Train, Validation, and Test metrics with toggleable legends.
    """
    if results_df.empty:
        logger.warning("No data available to plot metrics.")
        return

    save_dir = get_artifacts_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, f"toggleable_metrics_{Config.TIMESTAMP}.html")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Accuracy (Train, Validation, Test)", "Loss (Train, Validation, Test)"],
        vertical_spacing=0.1,
    )

    architectures = results_df["architecture"].unique()

    style_map = {
        "VanillaRNN": {"color": "blue", "dash": "solid"},
        "LSTM": {"color": "green", "dash": "dot"},
        "GRU": {"color": "red", "dash": "dashdot"},
    }

    for arch in architectures:
        arch_df = results_df[results_df["architecture"] == arch]

        for metric, row in [("accuracy", 1), ("loss", 2)]:
            fig.add_trace(
                go.Scatter(
                    x=arch_df["max_len"],
                    y=arch_df[f"train_{metric}"],
                    mode="lines+markers",
                    name=f"{arch} Train {metric.title()}",
                    line=dict(color=style_map[arch]["color"], dash=style_map[arch]["dash"]),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=arch_df["max_len"],
                    y=arch_df[f"val_{metric}"],
                    mode="lines+markers",
                    name=f"{arch} Validation {metric.title()}",
                    line=dict(color=style_map[arch]["color"], dash=style_map[arch]["dash"]),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=arch_df["max_len"],
                    y=arch_df[f"test_{metric}"],
                    mode="lines+markers",
                    name=f"{arch} Test {metric.title()}",
                    line=dict(color=style_map[arch]["color"], dash=style_map[arch]["dash"]),
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        title="Metrics Comparison Across Architectures (Toggleable Legends)",
        xaxis_title="Max Sequence Length",
        yaxis_title="Accuracy",
        yaxis2=dict(title="Loss"),
        height=900,
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
    )

    fig.write_html(save_file_path)
    logger.info(f"Toggleable metrics plot saved to {save_file_path}.")
