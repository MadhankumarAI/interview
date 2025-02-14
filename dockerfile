FROM python:3.11.7-slim  # Explicitly specify Python 3.11.7

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Verify Python version
RUN python --version

# Set up virtual environment explicitly with Python 3.11
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install setuptools first explicitly
RUN pip install --no-cache-dir setuptools setuptools-distutils wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8080
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:$PORT app:app
