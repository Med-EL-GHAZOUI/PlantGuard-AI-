"""Prétraitement d'une image isolée (utilisé par l'API à l'inférence), sans TensorFlow.

Choix de conception : l'image est renvoyée en float32 dans [0, 255], PAS normalisée.
La normalisation propre à chaque architecture (`preprocess_input` de Keras : [-1, 1] pour
MobileNetV2, soustraction de la moyenne ImageNet pour VGG16/ResNet50) sera placée DANS le
modèle. Ainsi l'entraînement et l'inférence appliquent exactement la même transformation.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ml.preprocessing.config import IMG_SIZE


def load_image(source: str | Path | bytes, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Charge une image (chemin ou octets), la met en RGB et la redimensionne.

    - applique l'orientation EXIF (les photos de téléphone sont souvent tournées) ;
    - convertit en RGB (gère PNG avec transparence, niveaux de gris, CMYK) ;
    - retourne un tableau (hauteur, largeur, 3) float32 dans [0, 255].

    Lève ValueError si le contenu n'est pas une image décodable.
    """
    try:
        fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        with Image.open(fp) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")
            # (largeur, hauteur) pour Pillow, alors que IMG_SIZE est (hauteur, largeur)
            img = img.resize((size[1], size[0]), Image.Resampling.BILINEAR)
            return np.asarray(img, dtype=np.float32)
    except Exception as exc:
        raise ValueError("Fichier image invalide ou illisible.") from exc


def to_batch(image: np.ndarray) -> np.ndarray:
    """Ajoute l'axe de lot : (H, W, 3) -> (1, H, W, 3), format attendu par `model.predict`."""
    return np.expand_dims(image, axis=0)
