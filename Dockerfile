FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY arena ./arena
COPY tasks.py inference.py models.py client.py __init__.py ./
COPY README.md ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openenv-core>=0.2.3" "fastapi" "uvicorn[standard]" "pydantic>=2" \
        "matplotlib" "numpy"

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn arena.server.app:app --host 0.0.0.0 --port ${PORT}"]
