# 🧠 ML Unsupervised Toolkit - Makefile

IMAGE_DEV = ml-unsupervised-dev
IMAGE_PROD = ml-unsupervised

# Build Docker images
docker-dev:
	docker build -t $(IMAGE_DEV) --target dev .

docker-prod:
	docker build -t $(IMAGE_PROD) --target prod .

# Run tests inside Dev image
test: docker-dev
	docker run --rm $(IMAGE_DEV) pytest tests --maxfail=1 --disable-warnings -q

# Run JupyterLab inside Dev image
notebook: docker-dev
	docker run -it -p 8888:8888 -v $(PWD):/app $(IMAGE_DEV) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Run Prod container (executes default CMD from Dockerfile)
run-prod: docker-prod
	docker run --rm $(IMAGE_PROD)

# Run with docker-compose (optional if you have a compose.yaml)
compose:
	docker-compose up
