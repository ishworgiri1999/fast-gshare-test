FROM --platform=${TARGETPLATFORM:-linux/amd64}  ghcr.io/openfaas/of-watchdog:0.9.6 as watchdog

FROM --platform=${TARGETPLATFORM:-linux/amd64} pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime as base

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Copy application code
COPY index.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV WHISPER_MODEL=small
ENV PRELOAD_MODEL=true
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000


COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
ENV fprocess="python /app/index.py"
CMD ["fwatchdog"]


