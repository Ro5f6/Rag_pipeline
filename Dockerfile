# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Keep the image lean and Python well-behaved in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /app

# Dependencies first, so a source-only change does not reinstall torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranking weights into the image.
#
# This is the single most important line in this file. Without it, every
# container start downloads ~500MB from Hugging Face before serving its first
# request, which turns an autoscaling event into a multi-minute outage and
# makes the service fail entirely in a network-restricted environment.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# Build the index at image build time so the first request is not the one
# paying for ingestion.
RUN python -m pipeline.orchestrator --ingest-only

EXPOSE 8000

# Readiness, not just liveness: the process is alive well before its models
# are loaded, and routing traffic to it in between produces timeouts.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
