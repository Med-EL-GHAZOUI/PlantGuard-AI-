"""Entraîne le CNN baseline et sauvegarde modèle, historique et métriques de VALIDATION.

Le jeu de test n'est volontairement PAS utilisé ici : il est réservé à la comparaison finale
(Phase 8) pour que la mesure reste honnête.

Usage :
    python -m ml.training.train_baseline --split-dir ml/dataset/split --epochs 10
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ml.preprocessing.class_info import compute_class_weights, count_images_per_class
from ml.preprocessing.config import SEED


def train(split_dir: Path, output_dir: Path, epochs: int, batch_size: int) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from ml.models.baseline_cnn import build_baseline_cnn
    from ml.preprocessing.data_pipeline import make_datasets

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, class_names = make_datasets(split_dir, batch_size=batch_size)
    weights = compute_class_weights(count_images_per_class(Path(split_dir) / "train"))

    model = build_baseline_cnn(len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor="val_accuracy",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=weights, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    summary = {
        "model": "baseline_cnn",
        "parameters": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
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
    args = parser.parse_args()
    print(json.dumps(train(args.split_dir, args.output_dir, args.epochs, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
