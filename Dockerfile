# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for numpy, scipy, hdbscan, umap)
RUN apt-get update && apt-get install -y     build-essential     python3-dev     libatlas-base-dev     gfortran     && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full repo
COPY . .

# Default command: run tests
CMD ["pytest", "tests/"]
