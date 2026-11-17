"""Construit les `tf.data.Dataset` train / validation / test à partir du dossier de split.

Règles :
- train : mélangé + augmenté ; validation et test : ni mélangés ni augmentés (mesure honnête) ;
- les images restent en [0, 255] (la normalisation sera dans le modèle, cf. preprocess.py) ;
- l'ordre des classes est alphabétique, identique à `class_names.json` et aux poids de classes.
"""
from __future__ import annotations

from pathlib import Path

from ml.preprocessing.augmentation import build_augmentation
from ml.preprocessing.config import BATCH_SIZE, IMG_SIZE, SEED


def make_datasets(split_dir: Path, batch_size: int = BATCH_SIZE, augment: bool = True):
    """Retourne (train_ds, val_ds, test_ds, class_names)."""
    import tensorflow as tf  # import paresseux

    split_dir = Path(split_dir)

    def load(name: str, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            split_dir / name,
            labels="inferred",
            label_mode="int",
            image_size=IMG_SIZE,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED,
        )

    train_ds = load("train", shuffle=True)
    class_names = train_ds.class_names
    val_ds = load("validation", shuffle=False)
    test_ds = load("test", shuffle=False)

    if augment:
        augmentation = build_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )
