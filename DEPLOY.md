# DomainAI Deployment Guide

This document provides instructions for deploying the DomainAI application using Docker.

## Prerequisites

- Docker and Docker Compose installed on your server
- At least 16GB RAM (recommended for running the model)
- GPU support is recommended but not required

## Deployment Steps

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd DomainAI
```

### 2. Build and Start the Docker Container

```bash
docker-compose up -d
```

This will:
- Build the Docker image using the provided Dockerfile
- Start the container in detached mode
- Map port 8000 on your host to port 8000 in the container
- Mount the fine_tuned_models directory as a volume

### 3. Verify the Deployment

Check if the API is running:

```bash
curl http://localhost:8000/health
```

You should receive a response like:

```json
{"status": "healthy", "model": "mistralai-Mistral-7B-v0-1-finetuned-domainai"}
```

### 4. Using the API

Make a request to the API:

```bash
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"business_description": "A modern coffee shop specializing in artisanal brews and pastries", "max_tokens": 32}'
```

## Additional Configuration

### Using a Different Model

To use a different fine-tuned model, update the MODEL_PATH environment variable in the docker-compose.yml file:

```yaml
environment:
  - MODEL_PATH=/app/fine_tuned_models/your-model-name
```

### Modifying Resource Allocation

If you need to adjust resource allocation for the container, update the deploy.resources section in docker-compose.yml.

## Troubleshooting

### Container Fails to Start

Check the container logs:

```bash
docker-compose logs domainai-api
```

### Model Loading Issues

Ensure the model path is correct and the model files are present in the fine_tuned_models directory.

## Stopping the Service

To stop the service:

```bash
docker-compose down