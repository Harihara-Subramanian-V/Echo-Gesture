# ==============================================================================
# EchoGesture Containerized Inference Environment
# Target: Python 3.10-slim on Linux x86_64
# Hackathon: Samsung R&D Institute India - XR Hackathon 2026 (Track 2)
# ==============================================================================

FROM python:3.10-slim

# Prevent interactive dialogs during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential system dependencies (ffmpeg, Mesa OpenGL, GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Copy dependency manifest
COPY docker/requirements.txt /app/requirements.txt

# Install pinned Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase and packaged assets
COPY . /app

# Default command: display predict help
CMD ["python", "predict.py", "--help"]
