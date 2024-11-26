import os
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.datasets import imdb


def get_statistics(data_lengths):
    """Compute summary statistics for sequence lengths."""
    mean = np.mean(data_lengths)
    median = np.median(data_lengths)
    max_len = np.max(data_lengths)
    q1 = np.percentile(data_lengths, 25)
    q2 = np.percentile(data_lengths, 50)
    q3 = np.percentile(data_lengths, 75)
    p95 = np.percentile(data_lengths, 95)
    p99 = np.percentile(data_lengths, 99)
    return mean, median, max_len, q1, q2, q3, p95, p99


def print_statistics(mean, median, max_len, q1, q2, q3, p95, p99):
    """Print summary statistics for sequence lengths."""
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Max: {max_len:.2f}")
    print(f"First Quartile (Q1 - 25th percentile): {q1:.2f}")
    print(f"Second Quartile (Q2 - 50th percentile - Median): {q2:.2f}")
    print(f"Third Quartile (Q3 - 75th percentile): {q3:.2f}")
    print(f"95th Percentile: {p95:.2f}")
    print(f"99th Percentile: {p99:.2f}")


def save_hist_and_quartiles_plotly(
    data_lengths,
    q1,
    q2,
    q3,
    p95,
    p99,
    save_filename: str = "sequence_lengths.html",
    save_dir: str = os.path.join(os.getcwd(), "eda", "plots"),
):
    """Plot histogram of sequence lengths with quartiles using Plotly."""
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
    print(f"Plot saved to {save_path}")


def compute_length_buckets(train_lengths, test_lengths):
    """
    Compute dynamic length buckets based on dataset statistics.

    Args:
        train_lengths: List or array of sequence lengths from the training set.
        test_lengths: List or array of sequence lengths from the test set.

    Returns:
        A list of tuples representing dynamic length buckets.
    """
    # Combine train and test lengths into a single dataset
    combined_lengths = np.concatenate([train_lengths, test_lengths])

    # Compute summary statistics dynamically
    _, _, max_len, q1, q2, q3, p95, p99 = get_statistics(combined_lengths)

    # Create dynamic length buckets based on these statistics
    length_buckets = [
        (0, int(q1)),  # 0 to Q1
        (int(q1), int(q2)),  # Q1 to Median
        (int(q2), int(q3)),  # Median to Q3
        (int(q3), int(p95)),  # Q3 to 95th Percentile
        (int(p95), int(p99)),  # 95th Percentile to 99th Percentile
        (int(p99), int(max_len)),  # 99th Percentile to Max
    ]

    return length_buckets


def main():
    # Load data with default settings
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    # Compute sequence lengths for training and test sets
    train_lengths = [len(seq) for seq in x_train]
    test_lengths = [len(seq) for seq in x_test]

    # Compute statistics for training and test sets
    train_mean, train_median, train_max, train_q1, train_q2, train_q3, train_p95, train_p99 = (
        get_statistics(train_lengths)
    )
    test_mean, test_median, test_max, test_q1, test_q2, test_q3, test_p95, test_p99 = (
        get_statistics(test_lengths)
    )

    # Print statistics
    print("Train Lengths Statistics:")
    print_statistics(
        train_mean, train_median, train_max, train_q1, train_q2, train_q3, train_p95, train_p99
    )
    print("\nTest Lengths Statistics:")
    print_statistics(
        test_mean, test_median, test_max, test_q1, test_q2, test_q3, test_p95, test_p99
    )

    # Compute dynamic buckets
    length_buckets = compute_length_buckets(train_lengths, test_lengths)
    print("\nNumber of Buckets:", len(length_buckets))

    # Print the number of sequences in each bucket
    for i, (start, end) in enumerate(length_buckets):
        bucket_count = sum(1 for length in train_lengths if start <= length < end)
        print(f"Bucket {i + 1}: {bucket_count} sequences")

    print("Dynamic Length Buckets:", length_buckets)

    save_hist_and_quartiles_plotly(
        train_lengths,
        train_q1,
        train_q2,
        train_q3,
        train_p95,
        train_p99,
        save_filename="train_sequence_lengths.html",
    )

    save_hist_and_quartiles_plotly(
        test_lengths,
        test_q1,
        test_q2,
        test_q3,
        test_p95,
        test_p99,
        save_filename="test_sequence_lengths.html",
    )


if __name__ == "__main__":
    main()
