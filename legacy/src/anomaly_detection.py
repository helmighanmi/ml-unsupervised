"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def run_isolation_forest(X, contamination=0.1, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    return labels, model

def run_oneclass_svm(X, nu=0.1, kernel="rbf", gamma="scale"):
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    labels = model.fit_predict(X)
    return labels, model

def run_lof(X, n_neighbors=20, contamination=0.1):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    return labels, model
