# DomainAI Inference API

This directory contains the inference API for the DomainAI domain name suggestion system. It provides both a REST API and a command-line interface for generating domain name suggestions based on business descriptions.

## Overview

The inference API uses the best-performing fine-tuned model (Mistral 7B) based on comprehensive evaluations performed in the DomainAI notebook. It exposes this functionality via:

1. **REST API**: A FastAPI server that can be deployed to provide domain name suggestions over HTTP
2. **CLI Tool**: A simple command-line script for quickly testing domain name suggestions

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### REST API

Start the API server:

```bash
cd DomainAI/inference
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 with the following endpoints:

- `GET /`: Check if the API is running
- `GET /health`: Health check for the API and model
- `POST /suggest`: Generate a domain name suggestion

Example API request:

```bash
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"business_description": "A modern coffee shop specializing in artisanal brews and pastries", "max_tokens": 32}'
```

### CLI Tool

Edit the variables at the top of the `cli.py` file to configure your request:

```python
# Business description to generate a domain name for
BUSINESS_DESCRIPTION = "A modern coffee shop specializing in artisanal brews and pastries"

# Model name to use for inference
MODEL_NAME = "mistralai-Mistral-7B-v0-1-finetuned-domainai"

# Maximum number of tokens to generate
MAX_TOKENS = 32

# Output format: "text", "json", or "pretty"
OUTPUT_FORMAT = "pretty"
```

Then run the script:

```bash
cd DomainAI/inference
python cli.py
```

## Directory Structure

- `__init__.py`: Package initialization
- `domain_inferencer.py`: Core inference functionality
- `api.py`: FastAPI server implementation
- `cli.py`: Command-line interface tool
- `requirements.txt`: Required dependencies

## Model Details

The default model used for inference is the fine-tuned Mistral 7B model, which was identified as the best-performing model in our evaluations. This model achieved:

- High relevance scores (3.992/5)
- Strong appropriateness scores (3.982/5)
- Good creativity scores (3.419/5)
- Best overall average score (3.798/5)