# Path: streamlit_app/app.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Thin Streamlit presentation layer for the unsupervised ML toolkit."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_unsupervised import AnomalyDetector, ClusteringPipeline, DimensionalityReducer
from ml_unsupervised.data import load_tabular_dataset
from ml_unsupervised.diagnostics import anomaly_summary, cluster_profile
from ml_unsupervised.exceptions import MLUnsupervisedError
from ml_unsupervised.visualization import plot_embedding


@st.cache_data(show_spinner=False)
def _load_iris() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    return iris.data.copy()


@st.cache_data(show_spinner=False)
def _load_upload(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(content)
        source = Path(handle.name)
    try:
        dataset = load_tabular_dataset(source)
    finally:
        source.unlink(missing_ok=True)
    return dataset.frame


def _dataset_picker() -> pd.DataFrame | None:
    st.sidebar.header("Dataset")
    use_default = st.sidebar.checkbox("Use Iris dataset", value=True)
    try:
        if use_default:
            return _load_iris()
        upload = st.sidebar.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
        if upload is None:
            st.info("Upload a dataset to continue.")
            return None
        return _load_upload(upload.getvalue(), upload.name)
    except (MLUnsupervisedError, ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return None


def _numeric_features(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=[np.number]).columns.tolist()


def _csv_download(df: pd.DataFrame, label: str, filename: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")


def _clustering_tab(frame: pd.DataFrame, features: list[str]) -> None:
    st.subheader("Clustering")
    if len(features) < 2:
        st.warning("Select at least two numeric features.")
        return
    method = st.selectbox(
        "Algorithm", ["kmeans", "dbscan", "hdbscan", "agglomerative", "gmm", "birch"], key="cl_method"
    )
    scaler = st.selectbox("Scaler", ["robust", "standard", "minmax"], key="cl_scaler")
    reduce = st.selectbox("Reduction", ["pca", "none"], key="cl_reduce")
    n_components = st.slider("Components (if reduced)", 2, min(5, len(features)), 2, key="cl_comp")
    params: dict[str, object] = {}
    if method in {"kmeans", "agglomerative", "birch"}:
        params["n_clusters"] = st.slider("Clusters (k)", 2, 10, 3, key="cl_k")
        if method == "kmeans":
            params.update({"random_state": 42, "n_init": 10})
    elif method == "gmm":
        params = {"n_components": st.slider("Components (k)", 2, 10, 3, key="cl_gmm_k"), "random_state": 42}
    elif method == "dbscan":
        params = {
            "eps": st.slider("eps", 0.1, 5.0, 0.8, 0.1, key="cl_eps"),
            "min_samples": st.slider("min_samples", 2, 20, 5, key="cl_ms"),
        }
    elif method == "hdbscan":
        params = {"min_cluster_size": st.slider("min_cluster_size", 2, 50, 10, key="cl_hdb")}

    if st.button("Run clustering", type="primary", key="cl_run"):
        try:
            X = frame[features].to_numpy(dtype=float)
            pipeline = ClusteringPipeline(
                scaler=scaler,
                clustering_method=method,
                clustering_params=params,
                reduction_method=None if reduce == "none" else reduce,
                n_components=n_components,
            )
            result = pipeline.fit_predict(X)
        except Exception as exc:  # noqa: BLE001 - show user-facing errors in UI
            st.error(str(exc))
            return

        out = frame.copy()
        out["cluster"] = result.labels
        cols = st.columns(3)
        cols[0].metric("Rows", len(out))
        cols[1].metric("Clusters", len(set(result.labels.tolist()) - {-1}))
        noise = float(np.mean(result.labels == -1))
        cols[2].metric("Noise ratio", f"{noise:.1%}")
        if result.metrics:
            st.json(result.metrics.to_dict())

        emb = result.transformed
        if emb.shape[1] >= 2:
            fig = plot_embedding(
                emb[:, : min(3, emb.shape[1])], labels=result.labels.astype(str), title="Cluster view"
            )
            st.plotly_chart(fig, use_container_width=True)

        prof = cluster_profile(frame[features], result.labels)
        st.markdown("**Cluster profile (feature means)**")
        st.dataframe(prof, use_container_width=True)
        _csv_download(out, "Download labeled data (CSV)", "clustered.csv")
        _csv_download(prof, "Download cluster profile (CSV)", "cluster_profile.csv")


def _anomaly_tab(frame: pd.DataFrame, features: list[str]) -> None:
    st.subheader("Anomaly detection")
    model_name = st.selectbox(
        "Algorithm", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"], key="an_alg"
    )
    contamination = st.slider("Expected anomaly ratio", 0.01, 0.5, 0.1, key="an_cont")
    projection_method = st.selectbox("Projection", ["PCA", "t-SNE", "UMAP"], key="an_proj")
    n_components = st.radio("Dimensions", [2, 3], horizontal=True, key="an_dim")
    method_map = {
        "Isolation Forest": "isolation_forest",
        "One-Class SVM": "one_class_svm",
        "Local Outlier Factor": "lof",
    }
    params: dict[str, object] = {}
    if model_name == "Isolation Forest":
        params = {"contamination": contamination, "random_state": 42}
    elif model_name == "One-Class SVM":
        params = {"nu": contamination, "kernel": "rbf", "gamma": "scale"}
    else:
        params = {
            "n_neighbors": st.slider("Neighbors", 5, 50, 20, key="an_nn"),
            "contamination": contamination,
        }

    if st.button("Detect outliers", type="primary", key="an_run"):
        try:
            X = frame[features].to_numpy(dtype=float)
            X_train, X_test = train_test_split(X, test_size=0.3, random_state=42) if len(X) > 10 else (X, X)
            detector = AnomalyDetector(method_map[model_name], params).fit(X_train)
            result = detector.predict(X_test)
            reducer_params = {}
            if projection_method == "t-SNE":
                perplexity = max(1.0, min(30.0, (len(X_test) - 1) / 3.0))
                if perplexity < len(X_test):
                    reducer_params = {"perplexity": perplexity}
            reducer = DimensionalityReducer(
                projection_method, n_components=n_components, params=reducer_params
            )
            projected = reducer.fit_transform(X_test)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        status = np.where(result.labels == 1, "Outlier", "Inlier")
        fig = plot_embedding(projected, labels=status, title=f"{projection_method} projection")
        st.metric("Detected outlier ratio", f"{result.outlier_ratio:.2%}")
        st.plotly_chart(fig, use_container_width=True)
        # Approximate feature-frame for displayed subset not tracked; show global summary on all features
        summary = anomaly_summary(pd.DataFrame(X_test, columns=features), result.labels)
        st.json(summary)
        if result.scores is not None:
            st.plotly_chart(
                px.histogram(pd.DataFrame({"anomaly_score": result.scores}), x="anomaly_score", nbins=30),
                use_container_width=True,
            )


def _reduction_tab(frame: pd.DataFrame, features: list[str]) -> None:
    st.subheader("Dimensionality reduction")
    method = st.selectbox(
        "Method", ["pca", "tsne", "umap", "kernel_pca", "ica", "nmf", "factor_analysis"], key="rd_method"
    )
    n_components = st.radio("Dimensions", [2, 3], horizontal=True, key="rd_dim")
    params: dict[str, object] = {}
    if method == "tsne":
        params = {"perplexity": st.slider("Perplexity", 2.0, 50.0, 30.0, key="rd_perp")}
    if st.button("Reduce", type="primary", key="rd_run"):
        try:
            X = frame[features].to_numpy(dtype=float)
            if method == "nmf" and np.any(X < 0):
                st.warning("NMF requires non-negative features; applying min-max shift for demo.")
                X = X - X.min(axis=0, keepdims=True)
            reducer = DimensionalityReducer(method, n_components=n_components, params=params)
            emb = reducer.fit_transform(X)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        fig = plot_embedding(emb, title=f"{method.upper()} embedding")
        st.plotly_chart(fig, use_container_width=True)
        out = frame.copy()
        for i in range(emb.shape[1]):
            out[f"component_{i + 1}"] = emb[:, i]
        _csv_download(out, "Download embedding (CSV)", "embedding.csv")


def main() -> None:
    st.set_page_config(page_title="Unsupervised ML Toolkit", layout="wide")
    st.title("Unsupervised ML Toolkit")
    st.caption(
        "Core ML logic is provided by the ml_unsupervised Python package; this app is a thin presentation layer."
    )

    frame = _dataset_picker()
    if frame is None:
        st.stop()
    if st.sidebar.checkbox("Show dataset preview", value=False):
        st.dataframe(frame.head(20), use_container_width=True)

    features = _numeric_features(frame)
    selected = st.sidebar.multiselect(
        "Features", features, default=features[: min(5, len(features))] or features
    )
    if not selected:
        st.warning("Select at least one feature.")
        st.stop()

    tabs = st.tabs(["Clustering", "Anomaly detection", "Reduction"])
    with tabs[0]:
        _clustering_tab(frame, selected)
    with tabs[1]:
        _anomaly_tab(frame, selected)
    with tabs[2]:
        _reduction_tab(frame, selected)


if __name__ == "__main__":
    main()
