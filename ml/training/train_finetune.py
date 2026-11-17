"""Fine-tuning (Phase 7) : repart du modèle à base gelée (Phase 6) et dégèle les dernières couches.

Learning rate très faible (1e-5) pour ne pas détruire les features ImageNet. Évalue la VALIDATION
uniquement ; le test est réservé à la Phase 8.

Usage :
    python -m ml.training.train_finetune --model mobilenetv2 --unfreeze 30 --epochs 5
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


def finetune(name: str, split_dir: Path, stage1_dir: Path, output_dir: Path,
             unfreeze: int, epochs: int, batch_size: int, lr: float) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from ml.models.transfer import unfreeze_top_layers
    from ml.preprocessing.data_pipeline import make_datasets

    checkpoint = stage1_dir / "best.keras"
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint} introuvable : lance d'abord ml.training.train_transfer.")
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, _ = make_datasets(split_dir, batch_size=batch_size)
    weights = compute_class_weights(count_images_per_class(Path(split_dir) / "train"))

    model = keras.models.load_model(checkpoint)
    n_unfrozen = unfreeze_top_layers(model, unfreeze)
    model.compile(optimizer=keras.optimizers.Adam(lr),
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
        "model": name, "stage": "finetuned", "unfrozen_layers": n_unfrozen,
        "learning_rate": lr, "parameters": int(model.count_params()),
        "trainable_parameters": trainable, "epochs_run": len(history.history["loss"]),
        "val_loss": round(float(val_loss), 4), "val_accuracy": round(float(val_acc), 4),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("ml/dataset/split"))
    parser.add_argument("--unfreeze", type=int, default=30, help="nombre de couches finales à dégeler")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    base = Path("ml/artifacts") / args.model
    result = finetune(args.model, args.split_dir, base, Path("ml/artifacts") / f"{args.model}_finetuned",
                      args.unfreeze, args.epochs, args.batch_size, args.lr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
