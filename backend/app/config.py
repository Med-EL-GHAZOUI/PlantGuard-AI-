"""Configuration lue dans les variables d'environnement (aucun secret dans le code)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("ml/artifacts/final")
    max_upload_bytes: int = 5 * 1024 * 1024
    database_url: str = "sqlite:///backend/plantguard.db"
    upload_dir: Path = Path("backend/uploads")
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5500", "http://127.0.0.1:5500"])

    @classmethod
    def from_env(cls) -> "Settings":
        default = cls()
        origins = os.getenv("PLANTGUARD_CORS_ORIGINS")
        return cls(
            model_dir=Path(os.getenv("PLANTGUARD_MODEL_DIR", str(default.model_dir))),
            database_url=os.getenv("PLANTGUARD_DATABASE_URL", default.database_url),
            upload_dir=Path(os.getenv("PLANTGUARD_UPLOAD_DIR", str(default.upload_dir))),
            max_upload_bytes=int(float(os.getenv("PLANTGUARD_MAX_UPLOAD_MB", "5")) * 1024 * 1024),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()] if origins
            else default.cors_origins,
        )
