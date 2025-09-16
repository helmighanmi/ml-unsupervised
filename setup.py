from setuptools import setup, find_packages

setup(
    name="ml-unsupervised",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "plotly",
        "hdbscan",
        "umap-learn",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8", "isort"]
    },
)
