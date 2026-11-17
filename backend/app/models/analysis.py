from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # SQLite ne stocke pas le fuseau : UTC naïf


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    image_filename: Mapped[str] = mapped_column(String(64), unique=True)  # nom généré (UUID), jamais celui de l'utilisateur
    class_name: Mapped[str] = mapped_column(String(128), index=True)
    plant: Mapped[str] = mapped_column(String(64), index=True)
    disease: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    is_healthy: Mapped[bool] = mapped_column(Boolean)
    low_confidence: Mapped[bool] = mapped_column(Boolean)
