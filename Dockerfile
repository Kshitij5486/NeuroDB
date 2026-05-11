# NeuroDB - Reinforcement Learning Query Optimizer
# Base image: Python 3.11 slim (smaller than 3.13, better compatibility)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements_docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=300 -r requirements_docker.txt

# Copy entire project
COPY . .

# Create directories
RUN mkdir -p runs models logs

# Expose port
EXPOSE 8000

# Environment variables with defaults
ENV PG_HOST=postgres
ENV PG_PORT=5432
ENV PG_DB=tpch
ENV PG_USER=postgres
ENV PG_PASSWORD=postgres123
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Startup script
CMD ["sh", "docker/entrypoint.sh"]