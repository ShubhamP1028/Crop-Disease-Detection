# Use Python 3.12 base image for consistency with your local setup
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY Requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r Requirements.txt

# Copy application files
COPY app.py .
COPY crop_disease_model.h5 .
COPY leaf-map.json .
COPY templates/ ./templates/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port your Flask app uses
EXPOSE 5030

# Set environment variables for production
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5030/health || exit 1

# Command to run the Flask app
CMD ["python", "app.py"]
