"""Compare des modèles entraînés sur le jeu de TEST (jamais vu à l'entraînement ni à la sélection).

Pour chaque modèle : accuracy, precision / recall / F1 macro (équitables malgré le déséquilibre
des classes), temps d'inférence par image, taille du fichier, nombre de paramètres, et matrice de
confusion. Tous les chiffres sont mesurés, jamais saisis à la main.

Usage :
    python -m ml.evaluation.compare_models --model baseline=ml/artifacts/baseline/best.keras --model mobilenetv2=ml/artifacts/mobilenetv2_finetuned/best.keras
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ml.preprocessing.config import IMG_SIZE


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy + précision/rappel/F1 macro et F1 pondéré (macro : chaque classe pèse pareil)."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "f1_weighted": float(f1_w),
    }


def measure_latency_ms(model, n_runs: int = 20) -> float:
    """Temps moyen (ms) pour prédire UNE image, après un échauffement (cible du cahier : < 2 s)."""
    x = np.random.default_rng(0).uniform(0, 255, (1, *IMG_SIZE, 3)).astype("float32")
    model.predict(x, verbose=0)  # échauffement : la 1re inférence inclut la compilation
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(x, verbose=0)
    return (time.perf_counter() - start) / n_runs * 1000


def predict_test_set(model, test_ds) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_pred.append(np.argmax(model.predict(images, verbose=0), axis=1))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def save_confusion_matrix(y_true, y_pred, class_names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # normalisée par classe réelle
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice de confusion normalisée — {path.stem.removeprefix('confusion_')}")
    if len(class_names) <= 40:
        ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_model(name: str, model_path: Path, test_ds, class_names: list[str],
                   output_dir: Path) -> dict:
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    y_true, y_pred = predict_test_set(model, test_ds)
    row = {
        "model": name,
        **compute_metrics(y_true, y_pred),
        "latency_ms": measure_latency_ms(model),
        "size_mb": model_path.stat().st_size / 1e6,
        "parameters": int(model.count_params()),
        "n_test_images": int(len(y_true)),
    }
    save_confusion_matrix(y_true, y_pred, class_names, output_dir / f"confusion_{name}.png")
    return row


def to_markdown(rows: list[dict]) -> str:
    head = ("| Modèle | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | "
            "Latence (ms/img) | Taille (Mo) | Paramètres |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    lines = [
        f"| {r['model']} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} | {r['recall_macro']:.4f} | "
        f"{r['f1_macro']:.4f} | {r['latency_ms']:.0f} | {r['size_mb']:.1f} | {r['parameters']:,} |"
        for r in sorted(rows, key=lambda r: -r["f1_macro"])
    ]
    return "\n".join([head, *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NOM=CHEMIN")
    parser.add_argument("--split-dir", type=Path, default=Path("ml/dataset/split"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    from ml.preprocessing.data_pipeline import make_datasets

    _, _, test_ds, class_names = make_datasets(args.split_dir, augment=False)
    rows = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Format attendu NOM=CHEMIN, reçu : {spec}")
        print(f"Évaluation de {name}...")
        rows.append(evaluate_model(name, Path(path), test_ds, class_names, args.output_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = to_markdown(rows)
    (args.output_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
