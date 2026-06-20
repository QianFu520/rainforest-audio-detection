FROM apache/airflow:2.9.3-python3.11

# Install system dependencies for audio processing
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user for pip installs
USER airflow

# Airflow AWS provider
RUN pip install --no-cache-dir \
    "apache-airflow-providers-amazon==8.25.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"

# PyTorch CPU-only — explicit index URL avoids pulling the 2GB CUDA build
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# TFLite inference — tensorflow-cpu is sufficient (tf.lite runs on CPU only)
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.17.0

# Audio + ML utilities
RUN pip install --no-cache-dir \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    numpy==2.3.5 \
    pandas==2.3.3

# Copy project source code and assets into the image
# Model weights are NOT baked in — mount outputs/models/ as a volume at runtime
WORKDIR /opt/rainforest-audio-detection
COPY src/ src/
COPY assets/ assets/
COPY config.py config.py
