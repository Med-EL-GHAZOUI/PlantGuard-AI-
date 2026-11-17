"""Phase 10 : exporte le modèle retenu avec sa configuration, prêt à être chargé par l'API.

Produit dans le dossier de sortie (défaut `ml/artifacts/final/`) :
- `model.keras`        : le modèle (prétraitement inclus : l'API envoie l'image brute en [0, 255]) ;
- `class_names.json`   : ordre des classes (indice -> nom) ;
- `model_config.json`  : métadonnées (taille d'entrée, nb de classes, seuil de confiance, empreinte
                         SHA-256, version de TensorFlow, métriques si fournies).

Les métriques ne sont jamais saisies à la main : elles sont lues depuis `comparison.json` (Phase 8).

Usage :
    python -m ml.export.export_model --model ml/artifacts/mobilenetv2_finetuned/best.keras --name mobilenetv2 --comparison docs/results/comparison.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ml.preprocessing.config import CLASS_NAMES_FILE, IMG_SIZE

MODEL_FILE = "model.keras"
CONFIG_FILE = "model_config.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_metrics(comparison_path: Path | None, name: str) -> dict | None:
    """Retrouve la ligne du modèle `name` dans comparison.json (None si absente ou non fournie)."""
    if comparison_path is None:
        return None
    rows = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    return next((row for row in rows if row["model"] == name), None)


def export_model(model_path: Path, class_names_path: Path, output_dir: Path, name: str,
                 comparison_path: Path | None = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
    """Valide la cohérence modèle / classes, copie les fichiers et écrit la configuration."""
    import tensorflow as tf
    from tensorflow import keras

    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold doit être strictement entre 0 et 1.")
    class_names = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
    model = keras.models.load_model(model_path)
    n_outputs = model.output_shape[-1]
    if n_outputs != len(class_names):
        raise ValueError(f"Le modèle prédit {n_outputs} classes mais class_names.json en liste "
                         f"{len(class_names)} : fichiers incohérents.")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / MODEL_FILE
    shutil.copy2(model_path, target)
    (output_dir / CLASS_NAMES_FILE).write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    config = {
        "name": name,
        "input_size": list(IMG_SIZE),
        "input_range": [0, 255],
        "n_classes": n_outputs,
        "confidence_threshold": confidence_threshold,
        "sha256": sha256_of(target),
        "tensorflow_version": tf.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_metrics": load_metrics(comparison_path, name),
    }
    (output_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--class-names", type=Path, default=Path("ml/artifacts") / CLASS_NAMES_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/final"))
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()
    config = export_model(args.model, args.class_names, args.output_dir, args.name,
                          args.comparison, args.confidence_threshold)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
