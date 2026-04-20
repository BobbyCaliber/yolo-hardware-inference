FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data

RUN apt update && apt install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt \
    && find /usr/local/lib/python3.10 -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.10 -type d -name 'tests' -prune -exec rm -rf {} + \
    && find /usr/local/lib/python3.10 -name '*.pyc' -delete

COPY check_cpu_config.py ./

ENTRYPOINT ["python3", "check_cpu_config.py"]
