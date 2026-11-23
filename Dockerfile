FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

WORKDIR /app

RUN groupadd --system mlops && useradd --system --gid mlops mlops

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY sql/ sql/
COPY src/ src/
COPY artifacts/mlops.db artifacts/mlops.db
COPY artifacts/registry/ artifacts/registry/

RUN chown -R mlops:mlops /app
USER mlops

EXPOSE 8080

CMD ["sh", "-c", "uvicorn cancer_mlops.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
