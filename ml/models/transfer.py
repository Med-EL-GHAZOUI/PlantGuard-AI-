"""Modèles Transfer Learning : VGG16, ResNet50, MobileNetV2 pré-entraînés sur ImageNet.

Phase 6 : base GELÉE (seule la nouvelle tête est entraînée). Le fine-tuning est la Phase 7.
La normalisation propre à chaque architecture est DANS le modèle sous forme d'une convolution 1x1
fixe (sérialisable sans Lambda) : l'API peut donc envoyer l'image brute en [0, 255].
"""
from __future__ import annotations

import numpy as np

from ml.preprocessing.config import IMG_SIZE

SUPPORTED = ("mobilenetv2", "resnet50", "vgg16")
_CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")


def _preprocessing_weights(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(noyau 3x3 d'entrée->sortie, biais) reproduisant `preprocess_input` de Keras."""
    if name == "mobilenetv2":  # [0,255] -> [-1,1]
        return np.eye(3, dtype="float32") / 127.5, np.full(3, -1.0, dtype="float32")
    # resnet50 / vgg16 : RGB -> BGR puis soustraction de la moyenne ImageNet
    return np.eye(3, dtype="float32")[::-1].copy(), -_CAFFE_MEAN_BGR


def _preprocessing_layer(name: str):
    from tensorflow import keras

    kernel, bias = _preprocessing_weights(name)
    layer = keras.layers.Conv2D(3, 1, name="preprocess", trainable=False)
    layer.build((None, *IMG_SIZE, 3))
    layer.set_weights([kernel.reshape(1, 1, 3, 3), bias])
    return layer


def build_transfer_model(name: str, num_classes: int, weights: str | None = "imagenet",
                         freeze_base: bool = True):
    """Construit base pré-entraînée + GlobalAveragePooling + Dropout + Dense softmax."""
    from tensorflow import keras

    if name not in SUPPORTED:
        raise ValueError(f"Modèle inconnu '{name}'. Choix : {', '.join(SUPPORTED)}")
    apps = keras.applications
    base_cls = {"mobilenetv2": apps.MobileNetV2, "resnet50": apps.ResNet50, "vgg16": apps.VGG16}[name]
    base = base_cls(include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3))
    base.trainable = not freeze_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = _preprocessing_layer(name)(inputs)
    # training=False : garde les statistiques BatchNorm de la base figées, même au fine-tuning
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name)
