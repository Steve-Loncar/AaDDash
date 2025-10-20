# Use a lightweight Python base
FROM python:3.11-slim

# Ensure output is unbuffered
ENV PYTHONUNBUFFERED=1

# Workdir
WORKDIR /app

# Install system deps (helpful for pandas/openpyxl, plotting and some image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python deps
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config to run in Docker
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false

# Start the app
CMD ["streamlit", "run", "app_rest.py", "--server.port=8501", "--server.headless=true"]