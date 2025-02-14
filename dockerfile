FROM python:3.9.13-slim

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

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway specific environment variable
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Start the application using gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app
