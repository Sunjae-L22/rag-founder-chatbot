FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=7860 HF_HOME=/app/.cache/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend /app/backend
COPY data/dataset.csv /app/data/dataset.csv
COPY web /app/web
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
