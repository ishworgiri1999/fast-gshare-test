# Makefile for building and running a Docker container

# Variables
IMAGE_NAME = ishworgiri/phi3-mini-onnxruntime
IMAGE_NAME_PYTORCH = ishworgiri/phi3-mini-pytorch

# Default target
all: build

# Build the Docker image with better compatibility
build:
	docker buildx build --platform linux/amd64 --progress=plain -t $(IMAGE_NAME) .

# Alternative build with native architecture (for Apple Silicon)
build-native:
	docker buildx build --platform linux/arm64 -t $(IMAGE_NAME) .

# Build with multi-architecture support
build-multi:
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE_NAME) .

# Build with no cache to ensure clean build
build-clean:
	docker buildx build --platform linux/amd64 --no-cache --progress=plain -t $(IMAGE_NAME) .

# Build using legacy builder (sometimes more compatible)
build-legacy:
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .

# PyTorch builds
build-pytorch:
	cd pytorch && docker buildx build --platform linux/amd64 --progress=plain -t $(IMAGE_NAME_PYTORCH) .

build-pytorch-native:
	cd pytorch && docker buildx build --platform linux/arm64 -t $(IMAGE_NAME_PYTORCH) .

# Run the container
run:
	docker run -p 5000:5000 -p 8080:8080 $(IMAGE_NAME)

# Stop and remove container
stop:
	docker stop $$(docker ps -q --filter ancestor=$(IMAGE_NAME)) || true

clean:
	docker rmi $(IMAGE_NAME) || true
	docker system prune -f

.PHONY: all build build-native build-multi build-clean build-legacy build-pytorch build-pytorch-native run stop clean