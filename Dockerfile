# ----------------------
# Stage 1: Dev / CI (with Jupyter & pytest)
# ----------------------
FROM python:3.10-slim AS dev

WORKDIR /app

# Install system dependencies (needed for numpy, scipy, hdbscan, umap)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (requirements + dev tools)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest jupyterlab

# Copy everything (for dev/testing)
COPY . .

# Run tests to validate build (only in dev/CI image)
RUN pytest tests --maxfail=1 --disable-warnings -q


# ----------------------
# Stage 2: Production (slim, no Jupyter)
# ----------------------
FROM python:3.10-slim AS prod

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files for runtime (no tests, no notebooks)
COPY src ./src
COPY config.yaml .
COPY README.md .

# Default command: run pipelines (can be overridden)
CMD ["python", "-m", "src.pipelines"]
