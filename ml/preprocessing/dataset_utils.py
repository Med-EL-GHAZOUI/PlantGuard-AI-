"""Fonctions utilitaires communes pour lire et vérifier un dataset d'images de feuilles.

Convention de nommage attendue (PlantVillage) : un dossier par classe, nommé
"<Plante>___<Maladie>", par exemple "Tomato___Early_blight" ou "Apple___healthy".
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_class_dirs(root: Path) -> list[Path]:
    """Retourne les sous-dossiers de `root` (un dossier = une classe), triés par nom."""
    return sorted(p for p in Path(root).iterdir() if p.is_dir())


def list_images(class_dir: Path) -> list[Path]:
    """Retourne les fichiers image (jpg/jpeg/png) d'un dossier de classe, triés."""
    return sorted(
        p for p in Path(class_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_class_name(class_name: str) -> tuple[str, str]:
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight'). Sans '___', plante = nom complet."""
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant, disease = class_name, "unknown"
    return plant.replace("_", " ").strip(), disease.replace("_", " ").strip()


def is_valid_image(path: Path) -> bool:
    """Vrai si Pillow sait réellement décoder l'image (détecte les fichiers corrompus)."""
    try:
        with Image.open(path) as img:
            img.verify()  # vérifie l'intégrité sans tout décoder
        with Image.open(path) as img:
            img.load()  # décodage complet : verify() seul rate certaines corruptions
        return True
    except Exception:
        return False


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash MD5 du contenu : sert à repérer les doublons exacts (risque de fuite de données)."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
