"""Télécharge automatiquement PlantVillage (Kaggle) dans ml/dataset/raw/.

Même principe que PFA/datasets/scripts/02_download_kaggle.py : on appelle la CLI Kaggle.
Prérequis : `pip install kaggle` et un token dans ~/.kaggle/ (kaggle.com > Settings > API).

Usage (depuis la racine du projet) :
    python -m ml.dataset.download_dataset

Le dataset « abdallahalidev/plantvillage-dataset » contient 3 variantes (color, grayscale, segmented).
On ne garde que `color` : images RGB originales, non augmentées.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
DATASET_DIR = Path(__file__).resolve().parent
RAW_DIR = DATASET_DIR / "raw"
TMP_DIR = DATASET_DIR / "_download"


def find_color_dir(root: Path) -> Path | None:
    """Cherche le dossier `color` (le nom exact peut varier d'un niveau selon l'archive)."""
    for candidate in root.rglob("color"):
        if candidate.is_dir():
            return candidate
    return None


def main() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"{RAW_DIR} contient déjà des données : téléchargement ignoré.")
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {KAGGLE_DATASET} (plusieurs centaines de Mo, patience)...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(TMP_DIR), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Échec du téléchargement. Vérifie : pip install kaggle, et le token dans ~/.kaggle/ "
            "(kaggle.com > Settings > API > Create New Token)."
        )

    color_dir = find_color_dir(TMP_DIR)
    if color_dir is None:
        raise SystemExit(f"Dossier 'color' introuvable dans {TMP_DIR} : structure inattendue.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(p for p in color_dir.iterdir() if p.is_dir()):
        shutil.move(str(class_dir), str(RAW_DIR / class_dir.name))
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"OK : {len(list(RAW_DIR.iterdir()))} classes dans {RAW_DIR}")


if __name__ == "__main__":
    main()
