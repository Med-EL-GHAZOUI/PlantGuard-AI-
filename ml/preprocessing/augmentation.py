"""Augmentation de données Keras, appliquée UNIQUEMENT au jeu d'entraînement.

Transformations choisies pour rester biologiquement plausibles :
- flips horizontal ET vertical : une feuille n'a pas de « haut » imposé ;
- rotation, zoom, translation légers : variations de cadrage d'une photo à l'autre ;
- luminosité / contraste modérés : variations d'éclairage.
Volontairement ABSENT : changement de teinte (hue). La couleur (jaunissement, taches brunes)
est le signal qui distingue les maladies ; la modifier détruirait l'information utile.
"""
from __future__ import annotations

from ml.preprocessing.config import SEED


def build_augmentation(seed: int = SEED):
    """Retourne un `keras.Sequential` de couches d'augmentation (actives seulement en training)."""
    from tensorflow import keras  # import paresseux : le reste du module marche sans TF

    layers = keras.layers
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.1, fill_mode="reflect", seed=seed),  # ±36°
            layers.RandomZoom((-0.15, 0.15), fill_mode="reflect", seed=seed),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
            layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
            layers.RandomContrast(0.15, seed=seed),
        ],
        name="augmentation",
    )
