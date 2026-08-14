# Needle 2 inference is a ctypes call into libneedle.so, so the server needs
# only huggingface_hub to fetch the engine and weights. cactus-needle declares
# jax/flax/optax/sentencepiece for training and export, which this image does
# not do -- installing it with --no-deps keeps the image near 200MB instead of
# well over 1GB. See README "Why --no-deps".
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps cactus-needle==2.0.3

COPY needle_openai ./needle_openai
COPY pyproject.toml README.md ./

# The engine (libneedle.so, ~14MB) and weights are pulled from HuggingFace on
# first start and cached here. Mount a volume to keep them across restarts.
RUN mkdir -p /cache/huggingface /cache/cactus-needle
VOLUME ["/cache"]

EXPOSE 8000

# Baked into the image so `docker run` needs no arguments; override with
# `docker run ... needle-openai --help`.
ENV NEEDLE_HOST=0.0.0.0 \
    NEEDLE_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r.get('status')=='ok' else 1)"

ENTRYPOINT ["python", "-m", "needle_openai"]


# --- optional: exact token counts + training extras -----------------------
# Adds sentencepiece so `usage` reports the model's real token counts.
FROM base AS full
RUN pip install --no-cache-dir sentencepiece
