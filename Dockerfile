ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# Stable for all backends. Do not declare TORCH_BACKEND above this block —
# BuildKit includes declared ARGs in the cache key of every later layer.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        git \
        ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && find /usr/local/lib -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf {} + || true \
    && rm -rf /root/.cache /tmp/*

ARG TORCH_BACKEND=cpu

# Chatterbox 0.1.7 pins torch==2.6.0 / torchaudio==2.6.0. Reinstall the matching
# CPU or CUDA wheel after the package install so the image backend is explicit.
RUN if [ "$TORCH_BACKEND" = "cpu" ]; then \
        pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip3 install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124; \
    fi \
    && rm -rf /root/.cache /tmp/*

ENV TTS_HOST=0.0.0.0 \
    TTS_PORT=8080 \
    TTS_VOICES=/config/voices.json \
    TTS_LANGUAGE=en \
    TTS_MODEL_NAME=tts-1 \
    TTS_VARIANT=turbo

COPY server.py device.py models.py adapter.py voices.example.json /app/
COPY static /app/static

EXPOSE 8080
VOLUME ["/config"]
CMD ["python3", "/app/server.py"]
