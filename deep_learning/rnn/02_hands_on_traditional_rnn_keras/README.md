
# IMDb Review Classification with Vanilla RNN and Hyperparameter Tuning

This project demonstrates how to classify IMDb movie reviews into positive or negative sentiments using a **Vanilla RNN (Simple RNN)** model in TensorFlow/Keras. Unlike more advanced RNN variants like LSTMs or GRUs, this project focuses on the traditional RNN architecture, showcasing its implementation, hyperparameter tuning, and performance visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Usage](#usage)
   - [Running the Pipeline](#running-the-pipeline)
6. [Features](#features)
7. [Results](#results)
8. [About the Author](#about-the-author)

---

## Overview

The project focuses on:
- Preprocessing the IMDb movie reviews dataset.
- Implementing a Vanilla RNN for binary classification.
- Optimizing model hyperparameters with **Keras Tuner**.
- Managing artifacts such as logs, plots, and model checkpoints.
- Visualizing results and hyperparameter tuning outcomes for interpretability.

The goal is to provide a comprehensive, educational pipeline for understanding and implementing traditional RNN models.

---

## Project Structure

The project files are organized as follows:

```
root/
├── config.py                 # Centralized configuration for hyperparameters and paths
├── data_preprocessing.py     # Functions for loading and preprocessing IMDb dataset
├── main.py                   # Main script to run the entire pipeline
├── model.py                  # Defines model architecture
├── tuner.py                  # Hyperparameter tuning logic
├── utils.py                  # Utility functions (e.g., logging, artifact management)
├── README.md                 # Project documentation
└── artifacts/                # Saved models, logs, plots, and results
```

---

## Requirements

To use this project, ensure you have the following installed:

- Python 3.8 or higher
- TensorFlow == 2.18.0
- Keras == 3.6.0
- Keras Tuner == 1.4.7
- Seaborn
- Pandas
- NumPy

Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate imdb-vanilla-rnn
```

---

## Usage

### Running the Pipeline

Run the main script to execute the full pipeline:

```bash
python main.py
```

Results will be saved in the `artifacts/` directory, including:
- Best model
- Training logs
- Hyperparameter tuning results
- Performance plots

---

## Features

- **Vanilla RNN Implementation**:
  - A Simple RNN architecture with an embedding layer, RNN units, and a dense output layer.

- **Hyperparameter Optimization**:
  - Optimize embedding dimensions, number of RNN units, and vocabulary size (`max_features`) using **Keras Tuner** with Hyperband.

- **Data Preprocessing**:
  - Tokenization and truncation for consistent input sizes using IMDb dataset.

- **Artifact Management**:
  - Automatically save logs, models, plots, and tuning results for reproducibility.

- **Visualization**:
  - Training and validation metrics over epochs.
  - Heatmaps of hyperparameter tuning results, highlighting the best trial.

---

## Results

1. **Best Hyperparameters**:
   The optimal hyperparameters are determined by Keras Tuner and saved in the logs and plots.

2. **Model Performance**:
   - Training, validation, and test accuracy are reported.
   - Results are visualized in learning curves and heatmaps.

3. **Visualization**:
   - **Heatmap**: A comprehensive view of all trials and hyperparameter combinations, with the best trial highlighted.

---

## About the Author

This project was created by **Irina Ryndova**, a Senior Data Scientist with expertise in machine learning, deep learning, and data-driven solutions.

- GitHub: [ryndovaira](https://github.com/ryndovaira)
- Email: [ryndovaira@gmail.com](mailto:ryndovaira@gmail.com)

Feel free to reach out for collaborations, questions, or feedback.

---

## Acknowledgements

- IMDb dataset from TensorFlow Datasets.
- TensorFlow and Keras for deep learning implementation.
- Keras Tuner for hyperparameter optimization.

---

## Notes for Further Improvements

This project uses a **Vanilla RNN** as a demonstration. While effective for small datasets, consider exploring **LSTM** or **GRU** models for better performance on more complex tasks or datasets.
