# Evolving Neural Networks with Particle Swarm Optimization

## Project Overview

This repository contains the work for a project for the **Artificial Intelligence** course at the Faculty of Electrical Engineering, University of Sarajevo.

The core objective of this project is to explore and demonstrate the effectiveness of using a metaheuristic optimization algorithm, **Particle Swarm Optimization (PSO)**, to "evolve" the hyperparameters of various neural networks. For each task, we establish a baseline model's performance and then compare it against a new model trained with hyperparameters discovered by PSO.

The project intentionally utilizes a wide variety of datasets to showcase how this methodology performs across different data types, complexities, and problem domains.

## Core Methodology

The workflow for each dataset follows a consistent three-step process:

1.  **Baseline Model:** A standard neural network (Dense or CNN) is created with sensible default hyperparameters, trained, and thoroughly evaluated on a test set. This establishes our performance benchmark.
2.  **PSO for Hyperparameter Optimization:** A PSO algorithm is configured to search a defined hyperparameter space (e.g., number of neurons/filters, learning rate, dropout). It evaluates the fitness of each "particle" (a set of hyperparameters) by training a temporary model and assessing its performance on a validation set.
3.  **Optimized Model & Comparison:** The best set of hyperparameters found by PSO is used to train a final, optimized model on the full training data. This model's performance is then compared against the baseline model across a comprehensive suite of metrics.

## Datasets & Tasks

A diverse set of six datasets was used to provide a broad analysis of the PSO methodology.

### Tabular Data (Classification & Regression)

*   **NASA Exoplanets (Classification):** A dataset used to classify planets as exoplanets or non-exoplanets based on astronomical measurements.
*   **S&P 500 (Regression):** A time-series dataset used to predict stock closing prices.
*   **Yu-Gi-Oh! Cards (Classification):** A custom-scraped dataset used to classify a monster card's "Attribute" (e.g., LIGHT, DARK, FIRE) based on its stats and type.
*   **Autolist Cars (Regression):** A custom-scraped dataset of used car listings, used to predict a car's `Price`.

### Complex Media (Audio & Image Classification)

*   **FMA: Free Music Archive (Audio Genre Classification):**
    *   **Task:** Classify 10-second audio clips into one of 8 balanced genres.
    *   **Data:** This project uses the `fma_small` and `fma_metadata` subsets. To replicate this work, you must download the data from the official [FMA GitHub repository](https://github.com/mdeff/fma). The audio files are preprocessed into Mel-spectrograms before being fed into a CNN.

*   **CIFAR-100 (Image Classification):**
    *   **Task:** Classify 32x32 color images into one of 100 object classes.
    *   **Data:** This is a standard computer vision benchmark dataset loaded directly via `tensorflow.keras.datasets`. It was chosen over the simpler CIFAR-10 to present a more significant challenge.

## Summary of Key Findings

This project successfully demonstrated that PSO is a versatile and powerful tool, but its impact and role change dramatically depending on the nature of the problem.

*   **PSO as a Refiner (NASA Exoplanets, CIFAR-100):** On well-structured problems where a strong baseline was already achievable, PSO acted as a fine-tuner. It discovered hyperparameter sets that provided consistent, though modest, improvements across all metrics, polishing an already good model into a slightly better one.

*   **PSO as a Problem-Solver (FMA Audio):** This was the most dramatic success case. To manage significant computational time on CPU resources and other time constraints, the input audio was shortened from 30s to 10s. This made the task much harder, and the baseline CNN completely failed to learn, performing at the level of random chance (12.5% accuracy). Despite these data limitations and a shortened PSO search, the optimizer was still able to discover a working set of hyperparameters (most notably a much smaller learning rate). This enabled the model to learn effectively, boosting its accuracy to **25.5%** and turning a failed model into a functional one.

*   **PSO Revealing Trade-offs (Autolist Cars):** On the car price regression task, the data contained significant outliers. Here, PSO found a "high-risk, high-reward" model. It achieved a *better* typical-case error (lower MAE and MedAE) than the baseline but performed *worse* on outlier-sensitive metrics (RMSE, R²). This highlights that the "best" model is context-dependent and that PSO can be used to optimize for different performance characteristics.

*   **PSO and Data Limitations (Yu-Gi-Oh! Cards):** On the heavily imbalanced Yu-Gi-Oh dataset, PSO provided only marginal gains. This result underscores a fundamental principle: while optimization can improve a model, its effectiveness is ultimately constrained by the inherent quality, balance, and predictive power of the dataset itself.

## How to Use This Repository

The Jupyter Notebooks (`.ipynb` files) contain the complete, commented code for data processing, model training, optimization, and evaluation for each of the four datasets handled in this part of the project.

-   To run the **CIFAR-100** notebook, no external data is needed.
-   To run the **FMA** notebook, you must first download the `fma_small.zip` and `fma_metadata.zip` files from the official FMA repository and place them in the project directory as described in the notebook.
-   The custom-scraped datasets (`autolist`, `yugioh`..) are included as `.json` files.

## Technologies Used

*   Python 3
*   TensorFlow & Keras
*   PySwarms
*   Scikit-learn
*   Pandas & NumPy
*   Librosa (for audio processing)
*   Matplotlib & Seaborn (for plotting)