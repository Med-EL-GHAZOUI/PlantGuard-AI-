"""Analyse d'un dataset d'images : corruption, comptage, déséquilibre, résolution, graphiques.

Usage :
    python -m ml.preprocessing.analyze_dataset --data-dir ml/dataset/raw --output-dir docs/figures
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on enregistre les figures dans des fichiers
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from ml.preprocessing.dataset_utils import (
    file_hash, is_valid_image, list_class_dirs, list_images, parse_class_name,
)

IMBALANCE_WARNING_RATIO = 3.0  # classe la plus grande / classe la plus petite
RESOLUTION_SAMPLE_PER_CLASS = 20
SEED = 42


def scan_dataset(data_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Parcourt le dataset. Retourne (tableau des images valides, liste des images corrompues)."""
    rows, corrupted = [], []
    for class_dir in list_class_dirs(data_dir):
        plant, disease = parse_class_name(class_dir.name)
        for path in list_images(class_dir):
            if is_valid_image(path):
                rows.append({"path": str(path), "class": class_dir.name,
                             "plant": plant, "disease": disease})
            else:
                corrupted.append(path)
    return pd.DataFrame(rows, columns=["path", "class", "plant", "disease"]), corrupted


def count_duplicates(df: pd.DataFrame) -> int:
    """Nombre d'images dont le contenu est identique à une autre (doublons exacts)."""
    if df.empty:
        return 0
    hashes = df["path"].map(lambda p: file_hash(Path(p)))
    return int(hashes.duplicated().sum())


def resolution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Résolutions et modes couleur sur un échantillon de RESOLUTION_SAMPLE_PER_CLASS par classe."""
    rng = random.Random(SEED)
    sizes = []
    for _, group in df.groupby("class"):
        paths = group["path"].tolist()
        for p in rng.sample(paths, min(RESOLUTION_SAMPLE_PER_CLASS, len(paths))):
            with Image.open(p) as img:
                sizes.append({"width": img.width, "height": img.height, "mode": img.mode})
    return pd.DataFrame(sizes)


def plot_class_distribution(counts: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(counts))))
    counts.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Nombre d'images")
    ax.set_title("Distribution des images par classe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_plant_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    df["plant"].value_counts().plot.bar(ax=ax)
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Images par plante")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze(data_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, corrupted = scan_dataset(data_dir)
    if df.empty:
        raise SystemExit(f"Aucune image valide trouvée dans {data_dir}")

    counts = df["class"].value_counts()
    ratio = counts.max() / counts.min()
    res = resolution_stats(df)
    duplicates = count_duplicates(df)

    plot_class_distribution(counts, output_dir / "class_distribution.png")
    plot_plant_distribution(df, output_dir / "plant_distribution.png")
    counts.rename_axis("class").reset_index(name="n_images").to_csv(
        output_dir / "class_counts.csv", index=False)

    summary = {
        "n_images_valid": len(df),
        "n_images_corrupted": len(corrupted),
        "n_classes": int(counts.size),
        "n_plants": int(df["plant"].nunique()),
        "min_class": f"{counts.idxmin()} ({counts.min()})",
        "max_class": f"{counts.idxmax()} ({counts.max()})",
        "imbalance_ratio": round(float(ratio), 2),
        "n_exact_duplicates": duplicates,
        "resolutions": sorted({f"{w}x{h}" for w, h in zip(res["width"], res["height"])})[:10],
        "color_modes": sorted(res["mode"].unique().tolist()),
    }
    print("=== RÉSUMÉ DU DATASET ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if corrupted:
        print("Images corrompues :", *[str(p) for p in corrupted[:20]], sep="\n  ")
    if ratio >= IMBALANCE_WARNING_RATIO:
        print(f"ATTENTION : déséquilibre (ratio {ratio:.1f}). Prévoir class_weight ou augmentation ciblée.")
    if duplicates:
        print(f"ATTENTION : {duplicates} doublons exacts. Le split les regroupe pour éviter les fuites.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Dossier avec un sous-dossier par classe")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    analyze(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
