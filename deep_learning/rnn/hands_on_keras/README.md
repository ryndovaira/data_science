
# IMDb Review Classification with RNN Architectures and Hyperparameter Tuning

This project demonstrates how to classify IMDb movie reviews into positive or negative sentiments using various RNN architectures (**Vanilla RNN**, **LSTM**, and **GRU**) in TensorFlow/Keras. The project includes dynamic dataset bucketing, hyperparameter tuning, logging, artifact management, and interactive visualization, providing an extensible and educational pipeline for machine learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
   - [Installation](#installation)
4. [Usage](#usage)
   - [Running the Pipeline](#running-the-pipeline)
5. [Features](#features)
6. [Results](#results)
   - [Timing and Performance](#timing-and-performance)
7. [Summary and Insights](#summary-and-insights)
8. [Contributing](#contributing)
9. [About the Author](#about-the-author)

---

## Overview

The project focuses on:
- Preprocessing the IMDb movie reviews dataset using configurable length buckets.
- Implementing multiple RNN architectures (**Vanilla RNN**, **LSTM**, and **GRU**) for binary classification.
- Optimizing model hyperparameters with **Keras Tuner**.
- Managing artifacts such as logs, plots, and model checkpoints.
- Visualizing results, training metrics, and hyperparameter outcomes.

This pipeline aims to provide both a professional portfolio example and an educational tool for learning deep learning concepts.

---

## Project Structure

The project files are organized as follows:

```
root/
├── config.py                 # Centralized configuration for hyperparameters and paths
├── data_preprocessing.py     # Functions for loading and preprocessing IMDb dataset
├── logger.py                 # Setup and configuration for logging
├── main.py                   # Main script to run the entire pipeline
├── plotter.py                # Visualization tools for metrics and results
├── saver.py                  # Utility functions for saving models and results
├── tuner.py                  # Hyperparameter tuning logic
├── utils.py                  # Utility functions (e.g., artifact management, checkpoints)
├── artifacts/                # Directory for logs, plots, final statistics, and checkpoints
│   ├── final_stats/          # Final metrics and results
│   ├── logs/                 # Log files
│   └── plots/                # Visualization of metrics
└── README.md                 # Project documentation
```

---

## Requirements

Ensure you have the following dependencies:

- Python == 3.12.7
- TensorFlow == 2.18.0
- Keras == 3.7.0
- Keras Tuner == 1.4.7
- Pandas == 2.2.3
- NumPy == 2.0.2
- Plotly == 5.24.1
- Scikit-learn == 1.5.2

### Installation

Use the provided `environment.yml` to create a controlled environment:

```bash
conda env create -f environment.yml
conda activate tensorflow_keras_cpu
```

> **Note**: This project uses the **CPU version** of TensorFlow. If you wish to use a GPU, you must install the GPU-compatible TensorFlow package separately:
> ```bash
> pip install tensorflow[and-cuda]
> ```

---

## Usage

### Running the Full Pipeline

Execute the main script to train and evaluate models across architectures and length buckets:

```bash
python main.py
```

Artifacts will be saved to the following locations:
- Training history: `artifacts/history/`
- Model checkpoints: `artifacts/checkpoints/`
- Visualizations: `artifacts/plots/`
- Logs: `artifacts/logs/`
- Final metrics and bucket results: `artifacts/final_stats/`

---

## Features

- **Dynamic Architecture Support**:
  - Choose between **Vanilla RNN**, **LSTM**, and **GRU** for training.
  - Configurable through the `Config` class in `config.py`.

- **Dynamic Length Bucketing**:
  - Preprocesses data into buckets of sequence lengths for more targeted model evaluation.
  - Includes sequence length statistics and visualizations.

- **Hyperparameter Optimization**:
  - Uses **Keras Tuner** to tune RNN units, embedding dimensions, and dropout rates.

- **Advanced Callbacks**:
  - Includes early stopping and learning rate reduction to prevent overfitting and optimize convergence.

- **Artifact Management**:
  - Automatically organizes logs, models, plots, and tuning results for reproducibility.

- **Interactive Visualization**:
  - Generate comparative accuracy and loss plots for architectures and length buckets.

---

## Results

### Timing and Performance

#### Configuration Parameters Affecting Time
1. **Sequence Length Buckets**:
   - Minimum Length: 130
   - Maximum Length: 175
2. **Epochs**: 50
3. **Batch Size**: 32
4. **Hyperparameter Tuning Epochs**: 15 (per configuration)
5. **Architecture**: Vanilla RNN, LSTM, GRU

#### Timing Overview
- **Total Time for All Experiments**: ~6 hours 45 minutes (NVIDIA GeForce RTX 4060)
- **Average Training Time Per Architecture**:
  - Vanilla RNN: ~2 hours 10 minutes
  - LSTM: ~2 hours 25 minutes
  - GRU: ~2 hours 10 minutes

Note: Time differences are primarily due to computational complexity differences between architectures.

---

## Summary and Insights

### Results Summary
- **Vanilla RNN**:
  - Performs well on smaller datasets but struggles with long-term dependencies.
  - Accuracy: ~85%
- **LSTM**:
  - Handles long-term dependencies effectively; performs best overall.
  - Accuracy: ~90%
- **GRU**:
  - Comparable to LSTM with fewer parameters; faster training.
  - Accuracy: ~89%

### Insights and Conclusions
1. **Accuracy vs. Complexity**:
   - LSTM provides the highest accuracy but is slightly slower.
   - GRU offers a balance between speed and performance.
   - Vanilla RNN is limited in capability but faster, suitable for simpler tasks.

2. **Resource Considerations**:
   - High-performance GPUs significantly reduce training time.
   - For non-GPU systems, consider reducing epochs or sample size.

3. **Future Work**:
   - Explore Transformer-based architectures for further improvements.
   - Use attention mechanisms to enhance interpretability.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## About the Author

This project was created by **Irina Ryndova**, a Senior Data Scientist passionate about crafting educational and robust machine learning pipelines.

- GitHub: [ryndovaira](https://github.com/ryndovaira)
- Email: [ryndovaira@gmail.com](mailto:ryndovaira@gmail.com)

Feel free to reach out for collaborations or feedback.

---
