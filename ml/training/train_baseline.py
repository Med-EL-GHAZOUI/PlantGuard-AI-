"""Entraîne le CNN baseline et sauvegarde modèle, historique et métriques de VALIDATION.

Le jeu de test n'est volontairement PAS utilisé ici : il est réservé à la comparaison finale
(Phase 8) pour que la mesure reste honnête.

Usage :
    python -m ml.training.train_baseline --split-dir ml/dataset/split --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from ml.preprocessing.class_info import compute_class_weights, count_images_per_class
from ml.preprocessing.config import SEED


def read_history(output_dir: Path) -> list[dict]:
    """Lignes déjà écrites dans history.csv (une par époque terminée), ou [] s'il n'existe pas."""
    path = output_dir / "history.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train(split_dir: Path, output_dir: Path, epochs: int, batch_size: int, resume: bool = False) -> dict:
    """Entraîne jusqu'à `epochs` époques AU TOTAL. Avec `resume`, repart de la dernière époque terminée."""
    import tensorflow as tf
    from tensorflow import keras

    from ml.models.baseline_cnn import build_baseline_cnn
    from ml.preprocessing.data_pipeline import make_datasets

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, class_names = make_datasets(split_dir, batch_size=batch_size)
    weights = compute_class_weights(count_images_per_class(Path(split_dir) / "train"))

    done = read_history(output_dir) if resume else []
    best_so_far = max((float(r["val_accuracy"]) for r in done), default=None)
    if resume:
        # last.keras (fin de la dernière époque) sinon best.keras ; le fichier contient aussi l'état de l'optimiseur
        source = next((p for p in (output_dir / "last.keras", output_dir / "best.keras") if p.exists()), None)
        if source is None or not done:
            raise SystemExit("Reprise impossible : aucun point de reprise (last.keras / best.keras) ou historique.")
        if len(done) >= epochs:
            raise SystemExit(f"Déjà {len(done)} époques terminées pour un objectif de {epochs} : rien à reprendre.")
        model = keras.models.load_model(source)
        print(f"Reprise depuis {source.name} après {len(done)} époques (meilleure val_accuracy : {best_so_far:.4f})")
    else:
        model = build_baseline_cnn(len(class_names))
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    callbacks = [
        # `initial_value_threshold` : une reprise ne doit pas écraser un meilleur modèle déjà sauvegardé
        keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor="val_accuracy",
                                        save_best_only=True, initial_value_threshold=best_so_far),
        keras.callbacks.ModelCheckpoint(output_dir / "last.keras"),  # point de reprise : fin de CHAQUE époque
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        keras.callbacks.CSVLogger(output_dir / "history.csv", append=resume),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=len(done),
              class_weight=weights, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    summary = {
        "model": "baseline_cnn",
        "parameters": int(model.count_params()),
        "epochs_run": len(read_history(output_dir)),
        "val_loss": round(float(val_loss), 4),
        "val_accuracy": round(float(val_acc), 4),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("ml/dataset/split"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/baseline"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true",
                        help="reprendre après la dernière époque terminée (--epochs = total visé)")
    args = parser.parse_args()
    print(json.dumps(train(args.split_dir, args.output_dir, args.epochs, args.batch_size, args.resume), indent=2))


if __name__ == "__main__":
    main()
