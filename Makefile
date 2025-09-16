# 🧠 ML Unsupervised Toolkit - Makefile

# Build Docker image
docker:
	docker build -t ml-unsupervised .

# Run tests inside Docker
test:
	docker run --rm ml-unsupervised

# Run Jupyter notebooks inside Docker
notebook:
	docker run -it -p 8888:8888 ml-unsupervised \
		jupyter notebook --ip=0.0.0.0 --allow-root

# Run with docker-compose
compose:
	docker-compose up
