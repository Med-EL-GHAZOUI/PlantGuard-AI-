"""Tests de la Phase 4 sur des images SYNTHÉTIQUES (aucune donnée réelle)."""
import io
import json

import numpy as np
import pytest
from PIL import Image

from ml.preprocessing.class_info import (
    compute_class_weights, count_images_per_class, save_class_names,
)
from ml.preprocessing.config import IMG_SIZE
from ml.preprocessing.preprocess import load_image, to_batch


def _png_bytes(mode="RGB", size=(300, 200), color=(10, 200, 30)):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_shape_dtype_range():
    arr = load_image(_png_bytes())
    assert arr.shape == (*IMG_SIZE, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 255.0


def test_load_image_non_square_uses_height_then_width():
    arr = load_image(_png_bytes(size=(300, 200)), size=(64, 96))
    assert arr.shape == (64, 96, 3)


def test_load_image_handles_grayscale_and_transparency():
    assert load_image(_png_bytes("L", color=128)).shape[-1] == 3
    rgba = load_image(_png_bytes("RGBA", color=(255, 0, 0, 0)))  # transparent -> fond blanc
    assert rgba.shape[-1] == 3 and rgba.min() == 255.0


def test_load_image_from_path(tmp_path):
    p = tmp_path / "leaf.png"
    p.write_bytes(_png_bytes())
    assert load_image(p).shape == (*IMG_SIZE, 3)


def test_load_image_rejects_garbage():
    with pytest.raises(ValueError):
        load_image(b"not an image")


def test_to_batch_adds_axis():
    assert to_batch(load_image(_png_bytes())).shape == (1, *IMG_SIZE, 3)


def test_class_weights_favor_small_classes():
    w = compute_class_weights({"A___x": 100, "B___y": 10, "C___z": 50})
    assert w[1] > w[2] > w[0]  # indices triés alphabétiquement : A=0, B=1, C=2
    # les poids « balanced » conservent le total : sum(w_i * n_i) == n_total
    assert sum(w[i] * n for i, n in enumerate([100, 10, 50])) == pytest.approx(160)


def test_class_weights_reject_empty_class():
    with pytest.raises(ValueError):
        compute_class_weights({"A___x": 5, "B___y": 0})


def test_class_names_saved_sorted(tmp_path):
    for cls in ("Tomato___blight", "Apple___healthy"):
        (tmp_path / "train" / cls).mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(tmp_path / "train" / cls / "a.jpg")
    assert count_images_per_class(tmp_path / "train") == {
        "Apple___healthy": 1, "Tomato___blight": 1}
    path = save_class_names(tmp_path / "train", tmp_path / "out")
    assert json.loads(path.read_text(encoding="utf-8")) == ["Apple___healthy", "Tomato___blight"]


def test_augmentation_keeps_shape_and_is_inactive_at_inference():
    pytest.importorskip("tensorflow")
    from ml.preprocessing.augmentation import build_augmentation

    x = np.random.default_rng(0).uniform(0, 255, (2, *IMG_SIZE, 3)).astype("float32")
    aug = build_augmentation()
    assert aug(x, training=True).shape == x.shape
    np.testing.assert_allclose(aug(x, training=False).numpy(), x)
