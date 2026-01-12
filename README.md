# Credit Card Fraud Detection

This project works on a Credit Card Fraud Detection dataset using Anomaly Detection techniques. It implements and compares two unsupervised learning algorithms: **Isolation Forest** and **Local Outlier Factor (LOF)** to identify fraudulent transactions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation and Usage](#installation-and-usage)
- [Methodology](#methodology)
- [Results](#results)

## Overview

The goal of this project is to detect fraudulent credit card transactions. Since valid transactions heavily mistu number fraud cases, this is an imbalanced classification problem. Instead of traditional supervised learning, we treat this as an outlier detection problem.

The script `main.py` performs the following steps:
1.  Loads the dataset (`creditcard.csv`).
2.  Performs exploratory data analysis (histograms, correlation heatmap).
3.  Preprocesses the data (scaling).
4.  Applies Isolation Forest and Local Outlier Factor algorithms.
5.  Evaluates the models using Accuracy, Precision, Recall, and F1-score.

## Dataset

The project requires a dataset named `creditcard.csv` in the root directory.
*   **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Assumed based on filename and context).
*   **Content**: The dataset typically contains transactions made by credit cards in September 2013 by European cardholders.
*   **Structure**: It contains numerical input variables (V1, V2, ... V28) which are the result of a PCA transformation, along with 'Time' and 'Amount'. The feature 'Class' is the target variable (1 in case of fraud and 0 otherwise).

## Prerequisites

The project requires **Python 3.x** and the following libraries:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`

## Installation and Usage

1.  **Clone the repository** (if applicable) or download the source code.
2.  **Install dependencies**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
    *(Note: It's recommended to use a virtual environment)*
3.  **Place the dataset**: Ensure `creditcard.csv` is located in the same directory as `main.py`.
4.  **Run the script**:
    ```bash
    python main.py
    ```

## Methodology

### algorithms
1.  **Isolation Forest**:
    *   Returns the anomaly score of each sample using the IsolationForest algorithm.
    *   It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

2.  **Local Outlier Factor (LOF)**:
    *   Measures the local deviation of density of a given data point with respect to its neighbors.
    *   It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood.

### Preprocessing
*   **Sampling**: The script samples 10% of the dataset for faster execution/prototyping.
*   **Scaling**: Features are scaled using `StandardScaler` to normalize the distribution, which is crucial for distance-based algorithms like LOF.

## Results

The script outputs specific metrics for each algorithm, including:
*   **Number of Errors**: Total misclassified samples.
*   **Accuracy Score**
*   **Precision, Recall, F1 Score**
*   **Classification Report**: Detailed breakdown of precision, recall, and f1-score for each class (Valid vs Fraud).

*Note: Since the dataset is highly imbalanced, Accuracy is not the best metric. Pay attention to Recall (for catching fraud) and Precision.*
