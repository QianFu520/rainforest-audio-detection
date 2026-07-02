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
# numpy pinned to <2.0 — PyTorch 2.2.2 was built before NumPy 2.0 and is incompatible with it
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    pandas==2.3.3

# MLflow skinny client — logging-only, no server/UI dependencies
# typing_extensions pinned: MLflow 3.x needs >=4.14.0; Airflow constraints would downgrade it
RUN pip install --no-cache-dir \
    "mlflow-skinny==3.13.0" \
    "typing_extensions>=4.14.0"

# Copy project source code and assets into the image
# Model weights are NOT baked in — mount outputs/models/ as a volume at runtime
WORKDIR /opt/rainforest-audio-detection
COPY src/ src/
COPY assets/ assets/
COPY config.py config.py
