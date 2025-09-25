# Use an official NVIDIA CUDA image as a base
# This provides the necessary CUDA toolkit and cuDNN libraries
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV GPU_MEMORY_FRACTION=""

# Install system dependencies, including Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -ms /bin/bash appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY --chown=appuser:appuser . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Copy the rest of the application code into the container


# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use uvicorn to run the FastAPI application
# Host 0.0.0.0 to make it accessible from outside the container
# You can limit GPU memory usage by setting the GPU_MEMORY_FRACTION environment variable (e.g., -e GPU_MEMORY_FRACTION=0.5)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
