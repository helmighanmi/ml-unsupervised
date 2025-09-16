# ----------------------
# Stage 1: Build & Test
# ----------------------
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies (needed for numpy, scipy, hdbscan, umap)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
 && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

# Copy source code and tests
COPY . .

# Run tests (will fail build if tests fail)
RUN pytest tests --maxfail=1 --disable-warnings -q


# ----------------------
# Stage 2: Runtime Image
# ----------------------
FROM python:3.10-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements (to leverage Docker layer caching)
COPY requirements.txt .

# Install only runtime dependencies (skip pytest)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (without tests)
COPY src ./src
COPY config.yaml .
COPY README.md .

# Default command (can be overridden at runtime)
CMD ["python", "-m", "src.pipelines"]
