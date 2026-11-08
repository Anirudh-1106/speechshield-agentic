# SpeechShield — headless moderation API image (text moderation + dashboard).
#
# Audio modes need host audio devices, which containers on Windows/macOS
# cannot reach; run those natively. On Linux you can pass /dev/snd or a
# PulseAudio socket through and switch --mode (see DEPLOYMENT.md).
#
#   docker build -t speechshield .
#   docker run -p 9501:9501 speechshield
#
# Build with --build-arg PRELOAD_MODELS=1 to bake the toxicity model in
# (bigger image, no download on first start).

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg portaudio19-dev gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && apt-get purge -y gcc libc6-dev && apt-get autoremove -y

COPY config ./config
COPY core ./core
COPY agents ./agents
COPY main.py dashboard.py ./

ARG PRELOAD_MODELS=0
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
AutoTokenizer.from_pretrained('unitary/toxic-bert'); M.from_pretrained('unitary/toxic-bert')"; \
    fi

RUN useradd --create-home --uid 1000 shield \
    && mkdir -p /app/models /app/logs \
    && chown -R shield:shield /app
USER shield

ENV SPEECHSHIELD_ENV=prod \
    SPEECHSHIELD_SERVER_HOST=0.0.0.0 \
    SPEECHSHIELD_CAPTURE_ENABLED=false \
    SPEECHSHIELD_DASHBOARD_ENABLED=false \
    SPEECHSHIELD_FILE_LOGGING=false

EXPOSE 9501
VOLUME ["/app/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9501/live', timeout=3).status == 200 else 1)"

CMD ["python", "main.py", "--mode", "api"]
