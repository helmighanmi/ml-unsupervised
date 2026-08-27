"""
Author: GHNAMI Helmi
Date: 2025-09-11
Position: Senior Data Scientist
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# ==========================
# CACHED DATA LOADERS
# ==========================
@st.cache_data(show_spinner=False)
def load_default_iris(test_size=0.3, random_state=42):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X


@st.cache_data(show_spinner=False)
def load_user_data(file, target_column, test_size=0.3, random_state=42):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload CSV or Excel.")
        return None, None, None, None, None

    if target_column not in df.columns:
        st.error(f"Column '{target_column}' not found in dataset.")
        return None, None, None, None, None

    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X


# ==========================
# DIMENSIONALITY REDUCTION
# ==========================
def reduce_dimensions(X, method_name, n_components):
    if method_name == "PCA":
        reducer = PCA(n_components=n_components)
    elif method_name == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method_name == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        return None
    return reducer.fit_transform(X)


# ==========================
# STREAMLIT APP
# ==========================
def main():
    st.set_page_config(page_title="Anomaly Detection App", layout="wide")
    st.title("🔍 Anomaly Detection Dashboard")
    st.caption("Created by Helmi Ghanmi — Senior Data Scientist")
    st.markdown("""
    A professional Streamlit app for **unsupervised anomaly detection** with interactive **dimensionality reduction** visualization.
    """)

    # -------------------------
    # SIDEBAR - Dataset selection
    # -------------------------
    st.sidebar.header("📁 Dataset Settings")
    use_default = st.sidebar.checkbox("Use Iris dataset (default)", value=True)

    if use_default:
        X_train, X_test, y_train, y_test, X_full = load_default_iris()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])
        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.sidebar.write("### Preview of Uploaded Dataset")
            st.sidebar.dataframe(df_preview.head())
            target_column = st.sidebar.selectbox("Select the target (Y) column:", df_preview.columns)
            X_train, X_test, y_train, y_test, X_full = load_user_data(uploaded_file, target_column)
        else:
            st.warning("Please upload a dataset to continue.")
            st.stop()

    # -------------------------
    # SIDEBAR - Model selection
    # -------------------------
    st.sidebar.header("⚙️ Model Parameters")
    model_name = st.sidebar.selectbox(
        "Choose an anomaly detection model:",
        ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
    )

    contamination = st.sidebar.slider("Contamination (expected anomaly ratio)", 0.01, 0.5, 0.1)

    if model_name == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42)
    elif model_name == "One-Class SVM":
        nu = st.sidebar.slider("Nu (upper bound on outliers)", 0.01, 0.5, 0.1)
        kernel = st.sidebar.selectbox("Kernel type", ["rbf", "linear", "poly"])
        model = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
    else:
        n_neighbors = st.sidebar.slider("Number of neighbors (LOF)", 5, 50, 20)
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)

    # -------------------------
    # SIDEBAR - Visualization
    # -------------------------
    st.sidebar.header("📉 Visualization Settings")
    dim_methods = st.sidebar.multiselect(
        "Select dimensionality reduction methods:",
        ["PCA", "t-SNE", "UMAP"],
        default=["PCA"]
    )

    n_components = st.sidebar.slider("Number of components (2D or 3D)", 2, 3, 2)
    show_data = st.sidebar.checkbox("Show dataset preview")

    # -------------------------
    # MAIN - Data display
    # -------------------------
    if show_data:
        st.subheader("Dataset Preview")
        st.dataframe(X_full.head())

    # -------------------------
    # DETECT BUTTON
    # -------------------------
    if st.button("🚀 Detect Outliers"):
        st.info(f"Running {model_name} on dataset...")

        # Fit and predict
        if model_name == "Local Outlier Factor":
            model.fit(X_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train)
            y_pred = model.predict(X_test)

        y_pred = np.where(y_pred == 1, 0, 1)  # 0=inlier, 1=outlier

        # ==========================
        # 1️⃣ Dimensionality Reduction Plots
        # ==========================
        if n_components == 2:
            st.subheader("🧭 2D Projections Comparison")
            n_methods = len(dim_methods)
            fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

            if n_methods == 1:
                axes = [axes]  # make iterable

            for i, method in enumerate(dim_methods):
                X_reduced = reduce_dimensions(X_test, method, n_components)
                inliers = X_reduced[y_pred == 0]
                outliers = X_reduced[y_pred == 1]
                axes[i].scatter(inliers[:, 0], inliers[:, 1], c="blue", label="Inliers", alpha=0.6, s=30)
                axes[i].scatter(outliers[:, 0], outliers[:, 1], c="red", label="Outliers", alpha=0.8, s=40)
                axes[i].set_title(f"{method} Projection (2D)")
                axes[i].legend()

            st.pyplot(fig, use_container_width=True)

        elif n_components == 3:
            for method in dim_methods:
                st.subheader(f"🧭 {method} 3D Projection")
                X_reduced = reduce_dimensions(X_test, method, n_components)
                df_3d = pd.DataFrame({
                    "x": X_reduced[:, 0],
                    "y": X_reduced[:, 1],
                    "z": X_reduced[:, 2],
                    "label": np.where(y_pred == 1, "Outlier", "Inlier")
                })
                fig_3d = px.scatter_3d(
                    df_3d,
                    x="x", y="y", z="z",
                    color="label",
                    color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                    title=f"{method} 3D Projection"
                )
                st.plotly_chart(fig_3d, use_container_width=True)

        # ==========================
        # 2️⃣ Anomaly Summary
        # ==========================
        st.subheader("📊 Anomaly Summary")

        outlier_ratio = np.mean(y_pred == 1)
        st.metric("Detected Outlier Ratio", f"{outlier_ratio * 100:.2f}%")

        if hasattr(model, "decision_function"):
            scores = -model.decision_function(X_test)
            fig_hist, ax = plt.subplots(figsize=(6, 4))
            ax.hist(scores, bins=30, color='skyblue', edgecolor='black')
            ax.set_title("Anomaly Score Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig_hist, use_container_width=True)
        else:
            st.info("Decision function not available for this model.")


if __name__ == "__main__":
    main()
