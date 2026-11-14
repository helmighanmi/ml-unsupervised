# Path: Dockerfile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install "."

COPY configs ./configs

USER app

# Default target: CLI-only production image (ml-unsupervised entrypoint).
ENTRYPOINT ["ml-unsupervised"]
CMD ["--help"]

# UI target: adds Streamlit and XLSX support for docker-compose / `--target ui`.
FROM runtime AS ui

USER root

RUN python -m pip install ".[ui]"

COPY streamlit_app ./streamlit_app

USER app

ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py"]
CMD ["--server.address=0.0.0.0", "--server.port=8501"]
