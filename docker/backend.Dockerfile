FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# couche séparée : les dépendances (lourdes) ne sont réinstallées que si requirements.txt change
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# seul le code nécessaire à l'inférence est copié (pas de dataset, d'entraînement ni de tests)
COPY backend backend
COPY ml/__init__.py ml/__init__.py
COPY ml/preprocessing ml/preprocessing

# utilisateur sans privilèges ; /data = base SQLite + images (volume)
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /data/uploads /models \
    && chown -R app /data
USER app

ENV PLANTGUARD_MODEL_DIR=/models \
    PLANTGUARD_DATABASE_URL=sqlite:////data/plantguard.db \
    PLANTGUARD_UPLOAD_DIR=/data/uploads

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "backend.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
