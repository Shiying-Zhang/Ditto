FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Ditto

COPY requirements.txt /app/Ditto/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/Ditto/requirements.txt

COPY . /app/Ditto

RUN python3 -m pip install -e /app/Ditto

ENV PYTHONPATH=/app/Ditto
ENV DITTO_MODELS_DIR=/models

CMD ["python3", "inference/infer_ditto.py", "--help"]
