"""Grad-CAM : montre quelles zones de la feuille ont influencé la prédiction.

Principe : on prend la dernière carte de caractéristiques convolutives (avant GlobalAveragePooling),
on calcule le gradient du score de la classe par rapport à cette carte, on moyenne ce gradient par
canal (importance du canal), puis on pondère la carte par ces importances. Le résultat, ramené à
[0, 1] et agrandi, est une carte de chaleur à superposer à l'image.

Fonctionne pour le CNN baseline comme pour les modèles Transfer Learning (base imbriquée) : on
coupe le modèle juste avant le premier GlobalAveragePooling2D.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ml.preprocessing.preprocess import load_image, to_batch


def _split_model(model):
    """Retourne (couches jusqu'à la carte conv, couches de tête à partir du GAP).

    Les couches sont appliquées en séquence (et non via un sous-modèle) : la base d'un modèle
    Transfer Learning est une couche imbriquée dont la sortie interne n'est pas reliée à l'entrée externe.
    """
    from tensorflow import keras

    gap_index = next((i for i, layer in enumerate(model.layers)
                      if isinstance(layer, keras.layers.GlobalAveragePooling2D)), None)
    if gap_index is None or gap_index == 0:
        raise ValueError("Grad-CAM : aucune couche GlobalAveragePooling2D trouvée dans le modèle.")
    body = [layer for layer in model.layers[:gap_index]
            if not isinstance(layer, keras.layers.InputLayer)]
    return body, model.layers[gap_index:]


def compute_heatmap(model, image: np.ndarray, class_index: int | None = None
                    ) -> tuple[np.ndarray, int, float]:
    """Grad-CAM d'une image (H, W, 3) float32 en [0, 255].

    Retourne (carte de chaleur dans [0, 1] de la taille de la carte conv, classe expliquée,
    probabilité de cette classe). Sans `class_index`, explique la classe prédite.
    """
    import tensorflow as tf

    body, head = _split_model(model)
    features = tf.convert_to_tensor(to_batch(image))
    for layer in body:
        features = layer(features, training=False)
    with tf.GradientTape() as tape:
        tape.watch(features)
        x = features
        for layer in head:
            x = layer(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(x[0]))
        score = x[:, class_index]
    grads = tape.gradient(score, features)
    channel_weights = tf.reduce_mean(grads, axis=(1, 2))              # (1, canaux)
    cam = tf.reduce_sum(features * channel_weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()                                      # seul l'influence positive compte
    peak = cam.max()
    heatmap = cam / peak if peak > 0 else np.zeros_like(cam)
    return heatmap.astype("float32"), class_index, float(score[0])


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Superpose la carte de chaleur (colormap « jet ») sur l'image d'origine."""
    from matplotlib import colormaps

    h, w = image.shape[:2]
    heat = Image.fromarray((heatmap * 255).astype("uint8")).resize((w, h), Image.Resampling.BICUBIC)
    colored = (colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255).astype("uint8")
    blended = (1 - alpha) * image + alpha * colored
    return Image.fromarray(blended.clip(0, 255).astype("uint8"))


def explain(model_path: Path, image_path: Path, output_path: Path,
            class_names: list[str] | None = None, class_index: int | None = None) -> dict:
    """Charge un modèle et une image, enregistre l'image superposée et retourne la classe expliquée."""
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    image = load_image(image_path)
    heatmap, idx, prob = compute_heatmap(model, image, class_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_heatmap(image, heatmap).save(output_path)
    label = class_names[idx] if class_names else str(idx)
    return {"class_index": idx, "class": label, "probability": round(prob, 4),
            "output": str(output_path)}


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Génère une image Grad-CAM.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/results/gradcam.png"))
    parser.add_argument("--class-names", type=Path, default=Path("ml/artifacts/class_names.json"))
    args = parser.parse_args()
    names = json.loads(args.class_names.read_text(encoding="utf-8")) if args.class_names.exists() else None
    print(json.dumps(explain(args.model, args.image, args.output, names), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
