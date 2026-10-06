# Virtual TES Gen0 — Production Multi-Service Container (Phase 5.9)
FROM python:3.12-slim

# Enforce clean unbuffered logs and bytecode optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy application source code
COPY app ./app

# Expose web service port
EXPOSE 8000

# Default command: FastAPI web service (worker overrides with: python -m app.worker)
CMD ["sh", "-c", "uvicorn app.web.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
