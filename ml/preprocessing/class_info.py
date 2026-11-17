"""Noms de classes et poids de classes (sans TensorFlow).

Le dataset est déséquilibré (ratio ~36, cf. docs/03-dataset.md) : sans correction, le modèle
favorise les grosses classes. `compute_class_weights` donne un poids plus fort aux petites.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml.preprocessing.config import CLASS_NAMES_FILE
from ml.preprocessing.dataset_utils import list_class_dirs, list_images


def count_images_per_class(split_dir: Path) -> dict[str, int]:
    """{nom_de_classe: nombre d'images} pour un dossier de split (ex. train/)."""
    return {d.name: len(list_images(d)) for d in list_class_dirs(split_dir)}


def compute_class_weights(counts: dict[str, int]) -> dict[int, float]:
    """Poids « balanced » : n_total / (n_classes * n_classe).

    Les clés sont les indices de classe (ordre alphabétique des noms, identique à celui de
    `image_dataset_from_directory`), format attendu par `model.fit(class_weight=...)`.
    """
    if not counts or any(n <= 0 for n in counts.values()):
        raise ValueError("Chaque classe doit contenir au moins une image.")
    total = sum(counts.values())
    n_classes = len(counts)
    return {i: total / (n_classes * counts[name]) for i, name in enumerate(sorted(counts))}


def save_class_names(split_dir: Path, output_dir: Path) -> Path:
    """Écrit la liste ordonnée des classes : l'API s'en sert pour convertir indice -> nom."""
    names = sorted(count_images_per_class(split_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CLASS_NAMES_FILE
    path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Écrit class_names.json et class_weights.json.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    counts = count_images_per_class(args.train_dir)
    save_class_names(args.train_dir, args.output_dir)
    weights = compute_class_weights(counts)
    (args.output_dir / "class_weights.json").write_text(
        json.dumps({str(k): round(v, 4) for k, v in weights.items()}, indent=2), encoding="utf-8")
    print(f"{len(counts)} classes ; poids min {min(weights.values()):.2f} / max {max(weights.values()):.2f}")


if __name__ == "__main__":
    main()
