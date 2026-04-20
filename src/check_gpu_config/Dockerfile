FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/usr/local/cuda/bin:${PATH}

RUN apt update && apt install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --target=/install -r /tmp/requirements.txt \
    && find /install -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /install -type d -name 'tests' -prune -exec rm -rf {} + \
    && find /install -name '*.pyc' -delete


FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data

RUN apt update && apt install -y --no-install-recommends \
        python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local/lib/python3.10/dist-packages

WORKDIR /app
COPY check_gpu_config.py ./

ENTRYPOINT ["python3", "check_gpu_config.py"]
