# Makefile for building and running a Docker container

# Variables
IMAGE_NAME = ishworgiri/phi3-mini-onnxruntime

IMAGE_NAME_PYTORCH= ishworgiri/phi3-mini-pytorch

# Default target
all: build

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

build-pytorch:
	cd pytorch && docker build -t $(IMAGE_NAME_PYTORCH) .

.PHONY: all build run stop rm clean