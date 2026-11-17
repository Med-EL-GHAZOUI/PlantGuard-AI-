"""Entraîne la tête d'un modèle Transfer Learning (base gelée) ; évalue la VALIDATION uniquement.

Usage :
    python -m ml.training.train_transfer --model mobilenetv2 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ml.models.transfer import SUPPORTED
from ml.preprocessing.class_info import compute_class_weights, count_images_per_class
from ml.preprocessing.config import SEED


def train(name: str, split_dir: Path, output_dir: Path, epochs: int, batch_size: int) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from ml.models.transfer import build_transfer_model
    from ml.preprocessing.data_pipeline import make_datasets

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, class_names = make_datasets(split_dir, batch_size=batch_size)
    weights = compute_class_weights(count_images_per_class(Path(split_dir) / "train"))

    model = build_transfer_model(name, len(class_names))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor="val_accuracy",
                                        save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=weights, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    summary = {
        "model": name, "stage": "frozen_base",
        "parameters": int(model.count_params()), "trainable_parameters": trainable,
        "epochs_run": len(history.history["loss"]),
        "val_loss": round(float(val_loss), 4), "val_accuracy": round(float(val_acc), 4),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("ml/dataset/split"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    out = args.output_dir or Path("ml/artifacts") / args.model
    print(json.dumps(train(args.model, args.split_dir, out, args.epochs, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
