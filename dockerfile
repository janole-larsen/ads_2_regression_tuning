# Image
FROM python:3.11-slim

# Working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    scikit-learn==1.6.1 \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    ydata-profiling \
    ipywidgets \
    jupyter \
    ipykernel \
    notebook

# Keep the container running
CMD ["sleep", "infinity"]
