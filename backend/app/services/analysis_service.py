"""Enregistrement et consultation des analyses (BF5, BF6, BF7). Toute la logique base de données est ici."""
from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.analysis import Analysis
from backend.app.services.prediction_service import Prediction

EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png"}


def save_analysis(session: Session, upload_dir: Path, image_bytes: bytes, image_format: str,
                  prediction: Prediction) -> Analysis:
    """Écrit l'image sous un nom UUID puis la ligne en base ; si la base échoue, retire l'image."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{EXTENSIONS[image_format]}"
    path = upload_dir / filename
    path.write_bytes(image_bytes)
    try:
        row = Analysis(
            image_filename=filename, class_name=prediction.class_name, plant=prediction.plant,
            disease=prediction.disease, confidence=prediction.confidence,
            is_healthy=prediction.is_healthy, low_confidence=prediction.low_confidence,
        )
        session.add(row)
        session.commit()
    except Exception:
        session.rollback()
        path.unlink(missing_ok=True)
        raise
    return row


def list_analyses(session: Session, limit: int, offset: int) -> tuple[int, list[Analysis]]:
    total = session.scalar(select(func.count(Analysis.id))) or 0
    rows = session.scalars(
        select(Analysis).order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit).offset(offset)
    ).all()
    return total, list(rows)


def get_analysis(session: Session, analysis_id: int) -> Analysis | None:
    return session.get(Analysis, analysis_id)


def delete_analysis(session: Session, upload_dir: Path, analysis_id: int) -> bool:
    """Supprime la ligne puis l'image. Retourne False si l'analyse n'existe pas."""
    row = session.get(Analysis, analysis_id)
    if row is None:
        return False
    filename = row.image_filename
    session.delete(row)
    session.commit()
    (upload_dir / filename).unlink(missing_ok=True)
    return True


def compute_stats(session: Session) -> dict:
    """Total, sains/malades, répartition par maladie, plantes les plus analysées, évolution par jour."""
    total = session.scalar(select(func.count(Analysis.id))) or 0
    healthy = session.scalar(select(func.count(Analysis.id)).where(Analysis.is_healthy.is_(True))) or 0
    by_disease = session.execute(
        select(Analysis.class_name, func.count(Analysis.id)).where(Analysis.is_healthy.is_(False))
        .group_by(Analysis.class_name).order_by(func.count(Analysis.id).desc(), Analysis.class_name)
    ).all()
    top_plants = session.execute(
        select(Analysis.plant, func.count(Analysis.id)).group_by(Analysis.plant)
        .order_by(func.count(Analysis.id).desc(), Analysis.plant).limit(5)
    ).all()
    per_day = session.execute(
        select(func.date(Analysis.created_at), func.count(Analysis.id))
        .group_by(func.date(Analysis.created_at)).order_by(func.date(Analysis.created_at))
    ).all()
    return {
        "total_analyses": total,
        "healthy_count": healthy,
        "diseased_count": total - healthy,
        "by_disease": [{"class_name": n, "count": c} for n, c in by_disease],
        "top_plants": [{"plant": n, "count": c} for n, c in top_plants],
        "per_day": [{"date": d, "count": c} for d, c in per_day],
    }
