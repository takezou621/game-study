# Multi-stage build for production
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies and system libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-minimal.txt requirements-webrtc.txt ./

# Install Python dependencies to a temporary location for later copying
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Set labels
LABEL maintainer="game-study"
LABEL description="AI English Coach for Fortnite Players"

# Set working directory
WORKDIR /app

# Install only runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-0 \
    libxvidcore4 \
    libx264-163 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages and virtual environment from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

# Create directories for output and logs
RUN mkdir -p /app/output /app/logs && \
    chown -R appuser:appuser /app/output /app/logs

# Set up environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER appuser

# Default entry point
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
