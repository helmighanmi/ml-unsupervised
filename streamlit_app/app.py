# Path: streamlit_app/app.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Thin Streamlit presentation layer for anomaly-detection workflows."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_unsupervised import AnomalyDetector, DimensionalityReducer


@st.cache_data(show_spinner=False)
def load_default_iris(test_size: float = 0.3, random_state: int = 42):
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    return X_train, X_test, X


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(content: bytes, filename: str, test_size: float = 0.3, random_state: int = 42):
    buffer = io.BytesIO(content)
    if filename.lower().endswith(".csv"):
        frame = pd.read_csv(buffer)
    elif filename.lower().endswith(".xlsx"):
        frame = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file type. Upload CSV or XLSX.")
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("The uploaded dataset does not contain numeric features.")
    if numeric.isna().any().any():
        raise ValueError("Numeric features contain missing values. Clean or impute them before running detection.")
    X_train, X_test = train_test_split(numeric, test_size=test_size, random_state=random_state)
    return X_train, X_test, frame


def main() -> None:
    st.set_page_config(page_title="Unsupervised ML Toolkit", layout="wide")
    st.title("Unsupervised Anomaly Detection")
    st.caption("Core ML logic is provided by the ml_unsupervised Python package; this page is presentation only.")

    st.sidebar.header("Dataset")
    use_default = st.sidebar.checkbox("Use Iris dataset", value=True)
    try:
        if use_default:
            X_train, X_test, preview = load_default_iris()
        else:
            upload = st.sidebar.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
            if upload is None:
                st.info("Upload a dataset to continue.")
                st.stop()
            X_train, X_test, preview = load_uploaded_dataset(upload.getvalue(), upload.name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Detector")
    model_name = st.sidebar.selectbox("Algorithm", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])
    contamination = st.sidebar.slider("Expected anomaly ratio", 0.01, 0.5, 0.1)

    method_map = {
        "Isolation Forest": "isolation_forest",
        "One-Class SVM": "one_class_svm",
        "Local Outlier Factor": "lof",
    }
    params: dict[str, object] = {}
    if model_name == "Isolation Forest":
        params = {"contamination": contamination, "random_state": 42}
    elif model_name == "One-Class SVM":
        params = {
            "nu": st.sidebar.slider("Nu", 0.01, 0.5, 0.1),
            "kernel": st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"]),
            "gamma": "scale",
        }
    else:
        params = {
            "n_neighbors": st.sidebar.slider("Neighbors", 5, 50, 20),
            "contamination": contamination,
        }

    st.sidebar.header("Projection")
    projection_method = st.sidebar.selectbox("Method", ["PCA", "t-SNE", "UMAP"])
    n_components = st.sidebar.radio("Dimensions", [2, 3], horizontal=True)

    if st.sidebar.checkbox("Show dataset preview"):
        st.dataframe(preview.head(20), use_container_width=True)

    if st.button("Detect outliers", type="primary"):
        detector = AnomalyDetector(method_map[model_name], params)
        detector.fit(X_train.to_numpy(dtype=float))
        result = detector.predict(X_test.to_numpy(dtype=float))

        reducer_params = {"perplexity": min(30, max(5, len(X_test) // 4))} if projection_method == "t-SNE" else {}
        reducer = DimensionalityReducer(projection_method, n_components=n_components, params=reducer_params)
        projected = reducer.fit_transform(X_test.to_numpy(dtype=float))

        projection = pd.DataFrame(projected, columns=[f"component_{i + 1}" for i in range(n_components)])
        projection["status"] = np.where(result.labels == 1, "Outlier", "Inlier")

        st.metric("Detected outlier ratio", f"{result.outlier_ratio:.2%}")
        if n_components == 2:
            fig = px.scatter(
                projection,
                x="component_1",
                y="component_2",
                color="status",
                title=f"{projection_method} projection",
            )
        else:
            fig = px.scatter_3d(
                projection,
                x="component_1",
                y="component_2",
                z="component_3",
                color="status",
                title=f"{projection_method} projection",
            )
        st.plotly_chart(fig, use_container_width=True)

        if result.scores is not None:
            score_frame = pd.DataFrame({"anomaly_score": result.scores})
            st.plotly_chart(px.histogram(score_frame, x="anomaly_score", nbins=30), use_container_width=True)


if __name__ == "__main__":
    main()
