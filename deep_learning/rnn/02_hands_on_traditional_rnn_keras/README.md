
# IMDb Review Classification with Vanilla RNN and Hyperparameter Tuning

This project demonstrates how to classify IMDb movie reviews into positive or negative sentiments using a **Vanilla RNN (Simple RNN)** model in TensorFlow/Keras. The project includes dynamic dataset bucketing, hyperparameter tuning, logging, artifact management, and interactive visualization, providing an extensible and educational pipeline for machine learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Usage](#usage)
   - [Running the Pipeline](#running-the-pipeline)
5. [Features](#features)
6. [Results](#results)
7. [Contributing](#contributing)
8. [About the Author](#about-the-author)

---

## Overview

The project focuses on:
- Preprocessing the IMDb movie reviews dataset using configurable length buckets.
- Implementing a Vanilla RNN for binary classification.
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
├── eda.py                    # Exploratory data analysis, sequence statistics, and plots
├── logger.py                 # Setup and configuration for logging
├── main.py                   # Main script to run the entire pipeline
├── model.py                  # Defines the model architecture (missing in files provided)
├── plotter.py                # Visualization tools for metrics and results
├── saver.py                  # Utility functions for saving models and results
├── tuner.py                  # Hyperparameter tuning logic
├── utils.py                  # Utility functions (e.g., artifact management, checkpoints)
└── README.md                 # Project documentation
```

---

## Requirements

To use this project, ensure you have the following installed:

- Python 3.8 or higher
- TensorFlow == 2.18.0
- Keras == 3.6.0
- Keras Tuner == 1.4.7
- Pandas, NumPy, Plotly, Seaborn, Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
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

- **Dynamic Length Bucketing**:
  - Preprocesses dataset into buckets of sequence lengths for targeted training and evaluation.
  - Facilitates insights into model performance across data variations.

- **Vanilla RNN Implementation**:
  - A Simple RNN model with tunable hyperparameters, including embedding dimensions and dropout.

- **Hyperparameter Optimization**:
  - Uses **Keras Tuner** to tune RNN units, embedding dimensions, and dropout rates.

- **Artifact Management**:
  - Automatically organizes logs, models, plots, and tuning results for reproducibility.

- **Visualization**:
  - Detailed training metrics and hyperparameter heatmaps.
  - Interactive plots for accuracy and loss trends using Plotly.

- **Logging**:
  - Centralized logging with both file and console output for debugging and analysis.

---

## Results

1. **Best Hyperparameters**:
   - Identified through Keras Tuner for each length bucket and saved for reproducibility.

2. **Training Metrics**:
   - Comprehensive performance evaluation with validation metrics and test accuracy.

3. **Visualization**:
   - Training history plots for loss and accuracy.
   - Comparative results across length buckets.

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

## Notes for Further Improvements

- Replace Vanilla RNN with LSTMs or GRUs for larger datasets or more complex tasks.
- Explore attention mechanisms for further accuracy gains.
