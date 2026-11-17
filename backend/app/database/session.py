"""Moteur SQLAlchemy et fabrique de sessions. SQLite par défaut ; PostgreSQL possible en changeant l'URL."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def create_session_factory(database_url: str) -> sessionmaker:
    """Crée le moteur, les tables manquantes, et retourne une fabrique de sessions."""
    from backend.app.models import analysis  # noqa: F401  (enregistre les tables dans Base)

    kwargs: dict = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}  # FastAPI utilise plusieurs threads
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool  # une seule connexion : sinon chaque thread aurait sa base vide
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
