"""Crée train/validation/test (70/15/15 par défaut), stratifiés par classe, sans fuite de données.

Anti-fuite : les images au contenu identique (même hash) sont regroupées et placées TOUTES dans le
même split. Sans cela, une même image pourrait se retrouver en train ET en test, ce qui gonflerait
artificiellement les résultats.

Usage :
    python -m ml.preprocessing.split_dataset --data-dir ml/dataset/raw --output-dir ml/dataset/split
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from ml.preprocessing.dataset_utils import (
    file_hash, is_valid_image, list_class_dirs, list_images,
)

SEED = 42
SPLITS = ("train", "validation", "test")


def group_by_content(paths: list[Path]) -> list[list[Path]]:
    """Regroupe les fichiers de contenu identique. Retourne une liste de groupes."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[file_hash(p)].append(p)
    return list(groups.values())


def split_groups(groups: list[list[Path]], train: float, val: float,
                 rng: random.Random) -> dict[str, list[Path]]:
    """Répartit les groupes dans les 3 splits selon les proportions (le reste va au test)."""
    rng.shuffle(groups)
    n = len(groups)
    n_train = round(n * train)
    n_val = round(n * val)
    # garantit au moins 1 groupe en validation et en test quand la classe est assez grande
    if n >= 3:
        n_val = max(n_val, 1)
        n_train = min(n_train, n - n_val - 1)
    parts = {
        "train": groups[:n_train],
        "validation": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }
    return {name: [p for g in gs for p in g] for name, gs in parts.items()}


def build_split(data_dir: Path, output_dir: Path, train: float = 0.70,
                val: float = 0.15) -> dict[str, dict[str, int]]:
    if not 0 < train < 1 or not 0 < val < 1 or train + val >= 1:
        raise ValueError("Proportions invalides : train + val doit être < 1 (le reste = test).")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} n'est pas vide : supprime-le ou choisis un autre dossier.")

    rng = random.Random(SEED)
    report: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for class_dir in list_class_dirs(data_dir):
        valid = [p for p in list_images(class_dir) if is_valid_image(p)]
        assignment = split_groups(group_by_content(valid), train, val, rng)
        for split, files in assignment.items():
            dest = output_dir / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
            report[split][class_dir.name] = len(files)
    return report


def check_no_leakage(output_dir: Path) -> int:
    """Retourne le nombre de contenus présents dans plus d'un split (doit valoir 0)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        for class_dir in list_class_dirs(output_dir / split):
            for p in list_images(class_dir):
                seen[file_hash(p)].add(split)
    return sum(1 for splits in seen.values() if len(splits) > 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    args = parser.parse_args()

    report = build_split(args.data_dir, args.output_dir, args.train, args.val)
    for split in SPLITS:
        print(f"{split}: {sum(report[split].values())} images")
    leaks = check_no_leakage(args.output_dir)
    print(f"Fuites détectées entre splits : {leaks}")
    if leaks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
