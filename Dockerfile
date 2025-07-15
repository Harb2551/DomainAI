# Use a base image with Python and PyTorch
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Set default model path environment variable
ENV MODEL_PATH=/app/fine_tuned_models_v4/mistralai-Mistral-7B-v0-1-finetuned-domainai

# Create directory for model caching
RUN mkdir -p /app/fine_tuned_models_v4

# Expose port for the API
EXPOSE 8000

# Set the default command to start the API server
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]