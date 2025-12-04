# Bank Marketing Machine Learning Analysis

This repository contains a machine learning project analyzing the UCI Bank Marketing dataset. The objective is to compare the performance of traditional machine learning models (Logistic Regression, Naive Bayes) with a feed-forward neural network implemented in PyTorch. The task is to predict whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related features.

## Project Overview

Financial institutions rely on targeted marketing to identify potential customers. This project evaluates how classical machine learning methods compare to a neural network in predicting customer subscription outcomes. The work includes exploratory data analysis, preprocessing, baseline model development, neural network training, and a final comparison of results.

## Repository Structure

    bank-marketing-ml/
    │
    ├── README.md
    ├── report/
    │   └── project_report.pdf
    │
    ├── data/
    │   ├── raw/
    │   │   └── bank.csv
    │   └── processed/
    │       └── bank_processed.csv
    │
    ├── notebooks/
    │   ├── 01_exploration.ipynb
    │   ├── 02_preprocessing.ipynb
    │   ├── 03_baseline_models.ipynb
    │   └── 04_neural_network.ipynb
    │
    ├── src/
    │   ├── preprocess.py
    │   ├── train_baseline.py
    │   ├── train_nn.py
    │   └── utils.py
    │
    ├── models/
    │   ├── logistic_regression.pkl
    │   ├── naive_bayes.pkl
    │   └── neural_net.pt
    │
    ├── results/
    │   ├── metrics/
    │   └── figures/
    │
    └── requirements.txt

## Methods

Baseline Models:
- Logistic Regression (scikit-learn)
- Naive Bayes (scikit-learn)

Neural Network:
- Fully connected feed-forward network implemented in PyTorch
- Standardization and feature encoding applied prior to training
- Trained using Adam optimizer and binary cross-entropy loss

## Evaluation

Models are compared using:
- Accuracy
- Precision, recall, and F1-score
- Confusion matrices
- ROC curves
- Training and validation loss (for the neural network)

The final report, located in the `report/` directory, provides full details, analysis, and discussion of results.

## How to Run

Install requirements:

    pip install -r requirements.txt

Run preprocessing:

    python src/preprocess.py

Train baseline models:

    python src/train_baseline.py

Train the neural network:

    python src/train_nn.py

## Data Source

Moro, Sérgio, Paulo Cortez, and Paulo Rita. “Bank Marketing Data Set.” UCI Machine Learning Repository, 2014.

## Source Code

All code used for data processing, model training, and evaluation is included in this repository.
