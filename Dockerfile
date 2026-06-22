FROM python:3.10-slim

# System libraries required by opencv-python-headless and DeepFace/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Cache directory for DeepFace's downloaded model weights
ENV DEEPFACE_HOME=/app/.deepface
RUN mkdir -p /app/.deepface

# Render/Railway inject PORT at runtime; default to 7860 for local testing
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
