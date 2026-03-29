FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential often needed for ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (to leverage Docker layer caching)
COPY requirements.txt .

# Install dependencies efficiently
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose default Streamlit port (mainly for local development)
EXPOSE 8501

# Command to run the application using $PORT environment variable for Railway
# Defaulting to 8501 if $PORT is not set
CMD streamlit run app/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
