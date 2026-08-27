# Path: Makefile
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

.PHONY: install install-all test lint format typecheck quality ci cli-help ui notebook docker-build docker-run

install:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[ui,nlp,notebooks,dev]"

test:
	pytest --cov=ml_unsupervised --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/ml_unsupervised

quality: lint typecheck test

ci: quality

cli-help:
	ml-unsupervised --help

ui:
	streamlit run streamlit_app/app.py

notebook:
	jupyter lab

docker-build:
	docker build -t ml-unsupervised:local .

docker-run:
	docker run --rm ml-unsupervised:local --help
