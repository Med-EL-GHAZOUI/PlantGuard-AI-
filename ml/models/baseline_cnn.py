"""CNN baseline construit from scratch : point de comparaison pour le Transfer Learning (Phases 6-8)."""
from __future__ import annotations

from ml.preprocessing.config import IMG_SIZE


def build_baseline_cnn(num_classes: int):
    """4 blocs Conv-BN-ReLU-MaxPool (32→256 filtres), pooling global, Dropout, softmax.

    La normalisation [0,255] -> [0,1] est DANS le modèle (cf. preprocess.py), donc l'API
    peut lui envoyer directement l'image brute. GlobalAveragePooling plutôt que Flatten :
    beaucoup moins de paramètres, donc moins de sur-apprentissage.
    """
    from tensorflow import keras

    layers = keras.layers
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")
