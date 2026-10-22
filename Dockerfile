FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SGX_PROJECT_ROOT=/app \
    TZ=Asia/Singapore
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 libxml2 libxslt1.1 libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock /app/requirements.lock
# CPU PyTorch is needed only for FinBERT during live ingestion. Set
# --build-arg INSTALL_TORCH=false for a review image without it.
ARG INSTALL_TORCH=true
RUN pip install -r /app/requirements.lock \
    && if [ "$INSTALL_TORCH" = "true" ]; then \
         pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu; fi
COPY sgx_capstone /app/sgx_capstone
COPY src /app/src
COPY prompts /app/prompts
COPY models /app/models
COPY config.yaml /app/config.yaml
COPY settings /app/settings
COPY outputs/research /app/outputs/research
COPY evidence /app/evidence
COPY tests /app/tests
RUN mkdir -p /app/outputs/report /app/outputs/sample_report /app/outputs/figures \
             /app/data /app/cache /app/raw_data /app/reports
CMD ["python", "-m", "sgx_capstone", "doctor"]
