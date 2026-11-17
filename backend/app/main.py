"""Point d'entrée FastAPI. Ne contient aucune logique métier (elle est dans services/).

Lancement : uvicorn backend.app.main:create_app --factory --reload
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.config import Settings
from backend.app.database.session import create_session_factory
from backend.app.security import SECURITY_HEADERS, SlidingWindowLimiter
from backend.app.services.knowledge_base import load_knowledge
from backend.app.services.prediction_service import ModelLoadError, PredictionService
from ml.preprocessing.config import CLASS_NAMES_FILE

logger = logging.getLogger("plantguard")


def create_app(settings: Settings | None = None,
               service: PredictionService | None = None) -> FastAPI:
    """`service` permet d'injecter un service déjà construit (tests) ; sinon on charge le modèle exporté."""
    settings = settings or Settings.from_env()
    app = FastAPI(title="PlantGuard AI", version="0.1.0",
                  description="Outil expérimental d'aide à l'identification des maladies de feuilles.",
                  docs_url="/docs" if settings.enable_docs else None,
                  redoc_url=None, openapi_url="/openapi.json" if settings.enable_docs else None)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            if name == "Cache-Control" and "cache-control" in response.headers:
                continue  # ne pas écraser le cache voulu d'un fichier servi (images)
            response.headers[name] = value
        return response

    app.state.settings = settings
    app.state.rate_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    app.state.knowledge = load_knowledge()
    app.state.session_factory = create_session_factory(settings.database_url)
    app.state.prediction_service = service
    app.state.model_name = None
    app.state.class_names = []

    if service is None:
        try:
            app.state.prediction_service = PredictionService.from_artifacts(settings.model_dir)
            config = json.loads((settings.model_dir / "model_config.json").read_text(encoding="utf-8"))
            app.state.model_name = config.get("name")
        except ModelLoadError as exc:
            logger.warning("Modèle non chargé : %s", exc)  # l'API démarre en mode dégradé (503)
    # la liste des classes reste disponible (/api/diseases) même si le modèle est absent
    names_file = settings.model_dir / CLASS_NAMES_FILE
    if app.state.prediction_service is not None:
        app.state.class_names = app.state.prediction_service.class_names
    elif names_file.exists():
        app.state.class_names = json.loads(names_file.read_text(encoding="utf-8"))

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Erreur non gérée", exc_info=exc)  # détail dans les logs, jamais au client
        return JSONResponse({"detail": "Erreur interne du serveur."}, status_code=500)

    app.include_router(router)
    return app

