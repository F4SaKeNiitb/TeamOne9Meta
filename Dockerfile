FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set PYTHONPATH so imports work
ENV PYTHONPATH=/app

# Enable the Gradio web UI at /web for manual testing
ENV ENABLE_WEB_INTERFACE=true

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
