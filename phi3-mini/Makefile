# Makefile for building and running a Docker container

# Variables
IMAGE_NAME = ishworgiri/phi3-mini-onnxruntime

# Default target
all: build

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .


.PHONY: all build run stop rm clean