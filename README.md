# DomainAI

A powerful domain name suggestion system that uses fine-tuned language models to generate creative, relevant, and appropriate domain names for businesses.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Usage](#cli-usage)
  - [API Usage](#api-usage)
- [API Reference](#api-reference)
  - [Suggest Domain Name](#suggest-domain-name)
  - [Health Check](#health-check)
- [Docker Deployment](#docker-deployment)
- [Advanced Configuration](#advanced-configuration)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for inference with full models)
- Docker and Docker Compose (for containerized deployment)

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DomainAI.git
cd DomainAI

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the fine-tuned model (if not using Docker)
# The model will be downloaded automatically when first using the inferencer
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DomainAI.git
cd DomainAI

# Build and start the containers
docker-compose up -d
```

## Usage

### CLI Usage

The CLI tool can be used by modifying the configuration variables in the `inference/cli.py` file:

1. Open the file and modify the following variables:
```python
# Business description to generate a domain name for
BUSINESS_DESCRIPTION = "a dating website which matches individuals with common interest"

# Model name to use for inference
MODEL_NAME = "mistralai-Mistral-7B-v0-1-finetuned-domainai"

# Maximum number of tokens to generate
MAX_TOKENS = 10

# Output format: "text", "json", or "pretty"
OUTPUT_FORMAT = "pretty"
```

2. Run the CLI tool:
```bash
# From the root directory
python -m inference.cli
```

The tool will output domain name suggestions based on the configured business description. The output format can be set to:
- `pretty`: Human-readable formatted output
- `json`: JSON output for programmatic use
- `text`: Simple text output with just the domain suggestions

### API Usage

Start the API server:

```bash
# Start the API server on port 8080
python -m inference.api
```

The API will be available at `http://localhost:8080`.

Example request using curl:

```bash
curl -X POST "http://localhost:8080/suggest" \
     -H "Content-Type: application/json" \
     -d '{"business_description":"A modern coffee shop specializing in artisanal brews and pastries located in San Francisco"}'
```

Example request using Python:

```python
import requests
import json

# Example request to suggest a domain name
response = requests.post(
    "http://localhost:8080/suggest", 
    json={
        "business_description": "A modern coffee shop specializing in artisanal brews and pastries located in San Francisco"
    }
)

# Parse the response
result = response.json()
print(f"Status: {result['status']}")
if result['status'] == "success":
    print(f"Suggested domains: {', '.join(result['suggestions'])}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Message: {result['message']}")
```

## API Reference

### Suggest Domain Name

Generates domain name suggestions for a business description.

**Endpoint:** `POST /suggest`

**Request Body:**

```json
{
  "business_description": "Your business description here"
}
```

**Success Response:**

```json
{
  "status": "success",
  "suggestions": ["domain1.com", "domain2.com", "domain3.com"],
  "confidence": 0.95
}
```

**Blocked Response (Edge Case):**

```json
{
  "status": "blocked",
  "suggestions": [],
  "message": "request contains inappropriate or ambiguous content"
}
```

**Error Response:**

```json
{
  "detail": "Error generating domain suggestion: [error message]"
}
```

### Health Check

Verifies that the API and model are running properly.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "model": "mistralai-Mistral-7B-v0-1-finetuned-domainai"
}
```

## Docker Deployment

The included Docker configuration provides a production-ready deployment with the following features:

1. Pre-loaded Mistral 7B fine-tuned model
2. API server exposed on port 8080
3. Optimized inference with GPU acceleration (when available)

```bash
# Start the Docker containers
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the containers
docker-compose down
```

## Advanced Configuration

### Model Selection

DomainAI supports three fine-tuned models:

- Mistral 7B (`mistralai-Mistral-7B-v0-1-finetuned-domainai`) - Recommended
- Llama 2 7B (`meta-llama-Llama-2-7b-hf-finetuned-domainai`)
- Qwen2 7B Instruct (`Qwen-Qwen2-7B-Instruct-finetuned-domainai`)

### Edge Case Detection

DomainAI automatically detects edge cases and inappropriate content. When such content is detected, the API returns a blocked response with status "blocked" and an empty suggestions list.

### Environment Variables

You can configure the API service using environment variables:

- `MODEL_NAME`: The model to use for inference (defaults to Mistral 7B)
- `PORT`: The port for the API server (default: 8080)
- `LOG_LEVEL`: Logging level (default: INFO)