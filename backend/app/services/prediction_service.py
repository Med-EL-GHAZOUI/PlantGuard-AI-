"""Phase 11 : service d'inférence. Charge le modèle UNE fois, puis prédit à partir d'octets d'image.

La logique métier vit ici (pas dans les routes FastAPI). Le prétraitement est celui de
`ml.preprocessing.preprocess` ; la normalisation propre au modèle est déjà dans le modèle.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ml.preprocessing.config import CLASS_NAMES_FILE
from ml.preprocessing.dataset_utils import parse_class_name
from ml.preprocessing.preprocess import load_image, to_batch

MODEL_FILE = "model.keras"
CONFIG_FILE = "model_config.json"


class ModelLoadError(RuntimeError):
    """Modèle ou configuration absents, corrompus ou incohérents."""


@dataclass(frozen=True)
class Prediction:
    plant: str
    disease: str
    class_name: str
    confidence: float
    is_healthy: bool
    low_confidence: bool
    top_k: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class PredictionService:
    def __init__(self, model, class_names: list[str], confidence_threshold: float = 0.6):
        n_outputs = model.output_shape[-1]
        if n_outputs != len(class_names):
            raise ModelLoadError(f"Le modèle prédit {n_outputs} classes, {len(class_names)} noms fournis.")
        self._model = model
        self._class_names = class_names
        self._threshold = confidence_threshold
        self._lock = threading.Lock()  # un seul appel au modèle à la fois (FastAPI = threads)
        self._model(np.zeros((1, *model.input_shape[1:]), dtype="float32"))  # échauffement

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> "PredictionService":
        """Charge `model.keras`, `class_names.json` et `model_config.json` d'un dossier exporté."""
        from tensorflow import keras

        artifacts_dir = Path(artifacts_dir)
        paths = {n: artifacts_dir / n for n in (MODEL_FILE, CLASS_NAMES_FILE, CONFIG_FILE)}
        missing = [n for n, p in paths.items() if not p.exists()]
        if missing:
            raise ModelLoadError(f"Fichiers manquants dans {artifacts_dir} : {', '.join(missing)}. "
                                 "Exporte d'abord le modèle : python -m ml.export.export_model")
        config = json.loads(paths[CONFIG_FILE].read_text(encoding="utf-8"))
        if _sha256(paths[MODEL_FILE]) != config["sha256"]:
            raise ModelLoadError("Empreinte du modèle différente de model_config.json : fichier modifié.")
        class_names = json.loads(paths[CLASS_NAMES_FILE].read_text(encoding="utf-8"))
        try:
            # safe_mode=True : refuse les couches Lambda / code arbitraire dans un fichier .keras
            model = keras.models.load_model(paths[MODEL_FILE], safe_mode=True)
        except Exception as exc:
            raise ModelLoadError(f"Chargement du modèle impossible ou refusé : {type(exc).__name__}") from exc
        return cls(model, class_names, config["confidence_threshold"])

    def predict(self, image_bytes: bytes, top_k: int = 3) -> Prediction:
        """Prédit la plante et la maladie. Lève ValueError si l'image est invalide."""
        batch = to_batch(load_image(image_bytes))
        with self._lock:
            probs = np.asarray(self._model(batch, training=False))[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"class_name": self._class_names[i], "confidence": float(probs[i])} for i in order]
        best = self._class_names[order[0]]
        plant, disease = parse_class_name(best)
        confidence = float(probs[order[0]])
        return Prediction(
            plant=plant, disease=disease, class_name=best, confidence=confidence,
            is_healthy=disease.lower() == "healthy",
            low_confidence=confidence < self._threshold, top_k=top,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
