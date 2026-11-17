"""Base de connaissances sur les maladies (F4) : fiches informatives, couverture partielle."""
from __future__ import annotations

import json
from pathlib import Path

from ml.preprocessing.dataset_utils import parse_class_name

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "diseases.json"


def load_knowledge(path: Path = DATA_FILE) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe_classes(class_names: list[str], knowledge: dict[str, dict]) -> list[dict]:
    """Une entrée par classe supportée ; description/conseil = None si aucune fiche n'existe."""
    entries = []
    for name in class_names:
        plant, disease = parse_class_name(name)
        info = knowledge.get(name, {})
        entries.append({
            "class_name": name, "plant": plant, "disease": disease,
            "is_healthy": disease.lower() == "healthy",
            "description": info.get("description"), "advice": info.get("advice"),
        })
    return entries
