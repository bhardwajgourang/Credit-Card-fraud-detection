import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))

# ---------- load data ----------
data = pd.read_csv('creditcard.csv')   # <-- ensure this path is correct
print("Loaded data shape:", data.shape)
print(data.columns.tolist())

# quick inspect
print(data.head())
print(data.info())

# sample 10% for faster prototyping (optional)
data = data.sample(frac=0.1, random_state=1).reset_index(drop=True)
print("After sampling 10% shape:", data.shape)

# histograms (optional)
data.hist(figsize=(20,20))
plt.tight_layout()
plt.show()

# count fraud / valid
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
# contamination should be fraction of outliers in TOTAL data
outlier_fraction = len(fraud) / float(len(data))
print("Outlier fraction (fraud / total):", outlier_fraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))

# correlation heatmap (optional)
corrmat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Prepare feature matrix X and target y
columns = [c for c in data.columns if c != 'Class']
X = data[columns].values
y = data['Class'].values
print("X shape:", X.shape, "y shape:", y.shape)

# Scale features (important for LOF)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# classifiers
state = 1
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X_scaled),
                                       contamination=outlier_fraction,
                                       random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                               contamination=outlier_fraction,
                                               novelty=False)  # use fit_predict below
}

n_outliers = len(fraud)

for clf_name, clf in classifiers.items():
    print("\n--- Running:", clf_name)
    if clf_name == "Local Outlier Factor":
        # LOF returns -1 for outliers and 1 for inliers
        y_pred = clf.fit_predict(X_scaled)
        # negative_outlier_factor_ is available after fit
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X_scaled)
        # decision_function: higher -> more normal for IsolationForest
        scores_pred = clf.decision_function(X_scaled)
        y_pred = clf.predict(X_scaled)
    
    # convert to 0 (valid) / 1 (fraud) labels consistently
    # both LOF and IsolationForest use 1 for inliers and -1 for outliers, but sklearn's IsolationForest.predict returns 1/-1
    y_pred = np.where(y_pred == 1, 0, 1)  # inliers -> 0, outliers -> 1
    
    # ensure same dtype and shape
    y_true = y.astype(int)
    assert y_pred.shape == y_true.shape
    
    n_errors = (y_pred != y_true).sum()
    print(f"Number of errors: {n_errors}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("Classification report:\n", classification_report(y_true, y_pred, zero_division=0))
