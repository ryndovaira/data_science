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
            text=[f"{val:.2f}" for val in history.history["loss"]],
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
            text=[f"{val:.2f}" for val in history.history["val_loss"]],
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
            text=[f"{val:.2f}" for val in history.history["accuracy"]],
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
            text=[f"{val:.2f}" for val in history.history["val_accuracy"]],
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


def plot_all_results(results_df):
    """
    Generate an interactive plot to compare accuracy and loss across architectures and length buckets using Plotly.
    """
    if results_df.empty:
        logger.warning("No data available to plot metrics.")
        return

    save_dir = get_artifacts_dir(Config.FINAL_STAT_DIR)
    save_file_path = os.path.join(save_dir, f"all_metrics_{Config.TIMESTAMP}.html")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
    )

    architectures = results_df["architecture"].unique()

    color_map = {
        "VanillaRNN": "blue",
        "LSTM": "green",
        "GRU": "red",
    }
    style_map = {
        "train": "dash",
        "val": "dot",
        "test": "solid",
    }

    for arch in architectures:
        arch_df = results_df[results_df["architecture"] == arch]

        for metric, row in [("accuracy", 1), ("loss", 2)]:
            for metric_type, dash_style in style_map.items():
                fig.add_trace(
                    go.Scatter(
                        x=arch_df["max_len"],
                        y=arch_df[f"{metric_type}_{metric}"],
                        mode="lines+markers+text",
                        text=arch_df[f"{metric_type}_{metric}"].round(2).astype(str),
                        textposition="top center",
                        name=f"{arch} {metric_type.title()} {metric.title()}",
                        line=dict(color=color_map[arch], dash=dash_style),
                        legendgroup=metric_type,
                    ),
                    row=row,
                    col=1,
                )
    fig.update_layout(
        title=dict(
            text="Metrics Comparison Across Architectures",
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        xaxis=dict(title="Max Sequence Length (Log Scale)", type="log"),
        yaxis=dict(
            title="Accuracy",
            range=[
                results_df[["train_accuracy", "val_accuracy", "test_accuracy"]].min().min() - 0.25,
                results_df[["train_accuracy", "val_accuracy", "test_accuracy"]].max().max() + 0.25,
            ],
        ),
        xaxis2=dict(title="Max Sequence Length (Log Scale)", type="log"),
        yaxis2=dict(
            title="Loss",
            range=[
                results_df[["train_loss", "val_loss", "test_loss"]].min().min() - 0.25,
                results_df[["train_loss", "val_loss", "test_loss"]].max().max() + 0.25,
            ],
        ),
        height=800,
        legend=dict(
            orientation="h",  # Horizontal legend
            xanchor="center",
            x=0.5,
        ),
    )

    fig.write_html(save_file_path)
    logger.info(f"Toggleable metrics plot saved to {save_file_path}.")


if __name__ == "__main__":
    # for debugging purposes
    import pandas as pd

    name = f"length_bucket_results_241127_190242.csv"
    df = pd.read_csv(os.path.join(get_artifacts_dir(Config.FINAL_STAT_DIR), name))
    plot_all_results(df)


def plot_hist_and_quartiles(
    data_lengths,
    q1,
    q2,
    q3,
    p95,
    p99,
    save_filename: str = "sequence_lengths.html",
    save_dir: str = get_artifacts_dir(Config.PLOT_DIR, "EDA"),
):
    """Plot histogram of sequence lengths with quartiles using Plotly."""
    logger.info("Saving histogram of sequence lengths with quartiles using Plotly.")
    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=data_lengths, nbinsx=50, marker_color="skyblue", opacity=0.7, name="Sequence Lengths"
        )
    )

    # Add vertical lines for quartiles with hover tooltips
    for value, label, color in zip(
        [q1, q2, q3, p95, p99],
        [
            "Q1 (25th percentile)",
            "Q2 (Median)",
            "Q3 (75th percentile)",
            "95th Percentile",
            "99th Percentile",
        ],
        ["blue", "green", "red", "purple", "orange"],
    ):
        fig.add_shape(
            type="line",
            x0=value,
            y0=0,
            x1=value,
            y1=1,
            line=dict(color=color, dash="dash"),
            xref="x",
            yref="paper",
        )
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[0],
                mode="markers",
                marker=dict(size=5, color=color),
                name=f"{label}: {value:0.0f}",
                hovertemplate=f"{label}: <b>{value:0.0f}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title="Sequence Length Distribution with Quartiles",
        xaxis_title="Sequence Length",
        yaxis_title="Frequency",
        template="plotly_white",
        bargap=0.1,
        height=600,  # Increase height to give space for annotations
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    fig.write_html(save_path)
    logger.info(f"Plot saved to {save_path}")
