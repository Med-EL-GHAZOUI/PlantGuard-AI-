"""Télécharge automatiquement PlantVillage (Kaggle) dans ml/dataset/raw/.

Même principe que PFA/datasets/scripts/02_download_kaggle.py : on appelle la CLI Kaggle.
Prérequis : `pip install kaggle six`. Le dataset public se télécharge sans token.

Usage (depuis la racine du projet) :
    python -m ml.dataset.download_dataset

Le dataset « abdallahalidev/plantvillage-dataset » contient 3 variantes (color, grayscale, segmented).
On extrait seulement `color` : images RGB originales, non augmentées. Si le zip est déjà
présent dans _download/ (téléchargement précédent interrompu avant l'extraction), il est réutilisé.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
DATASET_DIR = Path(__file__).resolve().parent
RAW_DIR = DATASET_DIR / "raw"
TMP_DIR = DATASET_DIR / "_download"
ZIP_PATH = TMP_DIR / "plantvillage-dataset.zip"


def download_zip() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {KAGGLE_DATASET} (plusieurs Go, patience)...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(TMP_DIR)],
        check=False,
    )
    if result.returncode != 0 or not ZIP_PATH.exists():
        raise SystemExit("Échec du téléchargement : vérifie ta connexion et `pip install kaggle six`.")


def extract_color_only(zip_path: Path, dest: Path) -> None:
    """Extrait uniquement les images du dossier `color/` vers dest/<classe>/<image>."""
    tmp_dest = dest.with_name(dest.name + "_partial")
    shutil.rmtree(tmp_dest, ignore_errors=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            # attendu : [..., "color", "<classe>", "<image>"]
            if info.is_dir() or "color" not in parts:
                continue
            idx = parts.index("color")
            if len(parts) != idx + 3:
                continue
            target = tmp_dest / parts[idx + 1] / parts[idx + 2]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    if count == 0:
        shutil.rmtree(tmp_dest, ignore_errors=True)
        raise SystemExit("Aucune image 'color/<classe>/<image>' trouvée dans le zip : structure inattendue.")
    # Renommage final : raw/ n'existe que si l'extraction est complète. Sous Windows, l'antivirus
    # ou l'indexeur peuvent retenir les fichiers quelques secondes : on réessaie.
    for attempt in range(12):
        try:
            tmp_dest.rename(dest)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(5)
    print(f"OK : {count} images extraites dans {dest}")


def main() -> None:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"{RAW_DIR} contient déjà des données : rien à faire.")
        return
    if not ZIP_PATH.exists() or ZIP_PATH.with_name(ZIP_PATH.name + ".kaggle-partial").exists() and \
            not zipfile.is_zipfile(ZIP_PATH):
        download_zip()
    extract_color_only(ZIP_PATH, RAW_DIR)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Dossier _download/ supprimé.")


if __name__ == "__main__":
    main()
