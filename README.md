
# Project Setup and Overview

Welcome to the **Data Science Project** repository. This project serves as a comprehensive example of deep learning and machine learning workflows, including dynamic data preprocessing, model training, and hyperparameter optimization.

It is organized to accommodate various sub-projects and experiments, such as RNN-based sentiment analysis for IMDb reviews.

---

## Setting Up the Environment

This guide explains how to set up and manage the Python environment for the project using `conda`. Follow these instructions to install dependencies, manage environments, and launch Jupyter Lab for interactive development.

### Commands

#### Create the Environment
Use the provided `environment.yml` to set up the project environment:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

Replace `<env_name>` with the desired name for your environment.

#### Export the Environment
To share your environment configuration (excluding builds):

```bash
conda env export --no-builds > environment.yml
```

This creates an `environment.yml` file that captures the environment's dependencies.

#### Install Additional Tools
Install the `nb_conda_kernels` package to manage Jupyter kernels within `conda` environments:

```bash
conda install -c conda-forge nb_conda_kernels
```

#### Launch Jupyter Lab
Start Jupyter Lab for working with notebooks:

```bash
jupyter-lab
```

---

## Finished Projects

### 1. RNN Sentiment Analysis with Keras
This sub-project demonstrates how to classify IMDb reviews using RNN architectures such as Vanilla RNN, LSTM, and GRU. It includes hyperparameter optimization, logging, and interactive visualizations.

- **Project Directory**: [deep_learning/rnn/hands_on_keras/](deep_learning/rnn/hands_on_keras/)
- **Detailed Documentation**: [hands_on_keras/README.md](deep_learning/rnn/hands_on_keras/README.md)

---

## Project Features

- **RNN Sentiment Analysis**: Explore the IMDb dataset using RNNs (Vanilla, LSTM, GRU) with dynamic data bucketing.
- **Hyperparameter Optimization**: Includes Keras Tuner for optimizing architectures and dropout rates.
- **Artifact Management**: Logs, visualizations, and model checkpoints are automatically organized.
- **Interactive Visualizations**: Training metrics and comparative results are plotted using Plotly.

---

For specific details about sub-projects, refer to their respective `README.md` files in subdirectories.
