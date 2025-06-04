# Makefile for building and running a Docker container

# Variables
IMAGE_NAME = ishworgiri/whisper


# Default target
all: build

# Build the Docker image
build:
	docker buildx build --platform=linux/amd64 -t $(IMAGE_NAME) .


.PHONY: all build run stop rm clean