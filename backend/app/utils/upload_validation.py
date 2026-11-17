"""Validation d'un fichier envoyé : extension, type MIME déclaré, taille, et contenu réel.

Les deux premiers contrôles sont déclaratifs (falsifiables) : le dernier ouvre vraiment le fichier
avec Pillow et vérifie son format, ce qui bloque un exécutable renommé en .jpg.
"""
from __future__ import annotations

import io
from pathlib import PurePath

from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class UploadValidationError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def validate_upload(filename: str | None, content_type: str | None, data: bytes,
                    max_bytes: int) -> None:
    """Lève UploadValidationError (400 / 413 / 415) si le fichier n'est pas acceptable."""
    if not data:
        raise UploadValidationError(400, "Le fichier est vide.")
    if len(data) > max_bytes:
        raise UploadValidationError(413, f"Fichier trop volumineux (maximum {max_bytes // (1024 * 1024)} Mo).")
    if PurePath(filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(415, "Extension non autorisée : utilisez JPG, JPEG ou PNG.")
    if (content_type or "").lower() not in ALLOWED_MIME_TYPES:
        raise UploadValidationError(415, "Type de fichier non autorisé : utilisez JPG, JPEG ou PNG.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.load()
    except Exception:
        raise UploadValidationError(400, "Le fichier n'est pas une image valide.") from None
    if fmt not in ALLOWED_FORMATS:
        raise UploadValidationError(415, "Format d'image non autorisé : utilisez JPG ou PNG.")
