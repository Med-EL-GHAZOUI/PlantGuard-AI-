"""Smoke test du CNN baseline : forme de sortie et un pas d'entraînement sur données synthétiques."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from ml.models.baseline_cnn import build_baseline_cnn  # noqa: E402
from ml.preprocessing.config import IMG_SIZE  # noqa: E402


def test_baseline_outputs_probabilities_and_trains():
    model = build_baseline_cnn(5)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    x = np.random.default_rng(0).uniform(0, 255, (4, *IMG_SIZE, 3)).astype("float32")
    y = np.array([0, 1, 2, 3])
    out = model.predict(x, verbose=0)
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-4)
    assert model.fit(x, y, epochs=1, verbose=0).history["loss"][0] > 0
