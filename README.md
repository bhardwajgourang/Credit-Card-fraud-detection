📌 Overview

This project implements an anomaly detection–based credit card fraud detection model using the popular Kaggle creditcard.csv dataset.
The dataset contains 284,807 transactions, where only 0.17% are fraudulent — making this a highly imbalanced classification problem.

To handle this, the project uses unsupervised anomaly detection models:

Isolation Forest

Local Outlier Factor (LOF)

Both models identify unusual patterns that differ significantly from normal transactions.

🚀 Features

Load and preprocess large credit card transaction data

Exploratory Data Analysis (EDA):

Histograms for each feature

Correlation heatmap

Data sampling (10% for faster prototyping)

Feature scaling using StandardScaler

Fraud detection using:

Isolation Forest

Local Outlier Factor

Performance evaluation with:

Accuracy

Precision

Recall

F1-score

Classification Report

Visualizations using Matplotlib and Seaborn

📂 Project Structure

📁 Credit-Card-Fraud-Detection
│

├── main.py

├── creditcard.csv        # (ignored by .gitignore; large dataset)

├── README.md

├── requirements.txt

└── .gitignore


⚙️ Installation

1️⃣ Clone this repository

git clone https://github.com/your-username/credit-card-fraud-detection.git

cd credit-card-fraud-detection

2️⃣ Create a virtual environment (recommended)

python3 -m venv venv

source venv/bin/activate     # macOS / Linux

venv\Scripts\activate        # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

🧠 Models Used
Isolation Forest
  Detects anomalies by “isolating” observations
  Works well on high-dimensional data
  Fast and scalable
Local Outlier Factor (LOF)
  Measures local deviation of density
  Flags points that differ significantly from neighbors
  More sensitive to feature scaling → requires StandardScaler
