FROM python:3.11-slim

LABEL maintainer="Terragon Labs <support@terragonlabs.ai>"
LABEL description="Spiking FPGA Toolchain - Production Container"
LABEL version="0.1.0-dev"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r spiking_fpga && useradd -r -g spiking_fpga spiking_fpga

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-dev.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Install the application
RUN pip install -e .

# Create directories for cache and logs
RUN mkdir -p /app/cache /app/logs && \
    chown -R spiking_fpga:spiking_fpga /app

# Switch to application user
USER spiking_fpga

# Expose default ports (if needed for web interface)
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app/src
ENV SPIKING_FPGA_CACHE_DIR=/app/cache
ENV SPIKING_FPGA_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from spiking_fpga import FPGATarget; print('OK')" || exit 1

# Default command
CMD ["spiking-fpga", "--help"]