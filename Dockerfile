# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set non-interactive installation and update apt sources
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support (adjust the extra-index-url if needed)
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install the other required Python packages
RUN pip install bitsandbytes && \
    pip install git+https://github.com/huggingface/transformers.git && \
    pip install git+https://github.com/huggingface/peft.git && \
    pip install git+https://github.com/huggingface/accelerate.git && \
    pip install datasets scipy trl huggingface_hub && \
    pip install pandas

# Set the working directory
WORKDIR /ws

# Copy the application code into the container
COPY . /ws

# Create a directory for results and mark it as a volume
RUN mkdir -p /ws/results
VOLUME ["/ws/results"]

# Set the default command to run your application
CMD ["/bin/bash"]
