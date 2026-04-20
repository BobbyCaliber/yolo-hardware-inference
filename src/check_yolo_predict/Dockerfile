FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    IMAGES_DIR=/app/data/images \
    YOLO_CONFIG_DIR=/app/weights \
    MPLBACKEND=Agg

RUN apt update && apt install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt \
    && find /usr/local/lib/python3.10 -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.10 -type d -name 'tests' -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.10 -name '*.pyc' -delete

COPY check_yolo_predict.py ./

RUN mkdir -p /app/weights /app/data
WORKDIR /app/weights

ENTRYPOINT ["python3", "/app/check_yolo_predict.py"]
