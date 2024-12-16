
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
   - [Future Work](#future-work)
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

#### Timing Overview from Logs
- **Vanilla RNN**: Total time ~147 seconds (~2 minutes 27 seconds); average per bucket ~29.4 seconds.
- **LSTM**: Total time ~2422 seconds (~40 minutes 22 seconds); average per bucket ~484.4 seconds (~8 minutes 4 seconds).
- **GRU**: Total time ~2450 seconds (~40 minutes 50 seconds); average per bucket ~489.9 seconds (~8 minutes 10 seconds).

#### Observations
- Vanilla RNN is the fastest across all buckets due to its simpler architecture.
- GRU and LSTM are comparably slower, with GRU slightly faster in some buckets.

### Bucket-Level Behavior

#### Accuracy by Bucket and Architecture
- **Vanilla RNN**:
  - Bucket 0-129: ~78.6% test accuracy.
  - Bucket 130-175: ~78.4% test accuracy.
  - Bucket 176-284: ~77.0% test accuracy.
  - Bucket 285-597: ~55.8% test accuracy.
  - Bucket 598-2494: ~53.3% test accuracy.

- **LSTM**:
  - Bucket 0-129: ~90.1% test accuracy.
  - Bucket 130-175: ~88.7% test accuracy.
  - Bucket 176-284: ~87.9% test accuracy.
  - Bucket 285-597: ~85.2% test accuracy.
  - Bucket 598-2494: ~83.4% test accuracy.

- **GRU**:
  - Bucket 0-129: ~89.8% test accuracy.
  - Bucket 130-175: ~87.5% test accuracy.
  - Bucket 176-284: ~86.7% test accuracy.
  - Bucket 285-597: ~84.1% test accuracy.
  - Bucket 598-2494: ~82.3% test accuracy.

#### Key Takeaways
- **Short Sequences**: All architectures achieve higher accuracy on shorter sequences, with LSTM and GRU exceeding 85% consistently.
- **Long Sequences**: Vanilla RNN struggles with longer sequences (>285 tokens) due to lack of memory capabilities, whereas LSTM and GRU handle these better, retaining ~82-85% accuracy.

---

## Summary and Insights

### Results Summary
- **Vanilla RNN**: Best suited for short sequences but faces a steep drop in accuracy for longer buckets.
- **LSTM**: Excels at long sequences, offering the highest accuracy consistently across buckets.
- **GRU**: A balanced choice, trading a slight loss in accuracy for faster training times.

### Future Work
1. **Transformer-Based Models**:
   - Investigate the use of **Transformers**, **BERT**, and **LLaMA** for improved performance and interpretability.
   - Incorporate pre-trained models like **ChatGPT** for better context understanding.

2. **Attention Mechanisms**:
   - Enhance RNN models with attention layers to improve accuracy for longer sequences.

3. **Expand the Dataset**:
   - Test on larger, more diverse datasets to generalize the findings.

4. **Explainable AI**:
   - Explore explainability techniques to understand feature importance and decision-making processes.

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
