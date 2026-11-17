"""Tests Phase 6 sans télécharger ImageNet (weights=None) : structure, gel, prétraitement."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras  # noqa: E402

from ml.models.transfer import SUPPORTED, build_transfer_model  # noqa: E402
from ml.preprocessing.config import IMG_SIZE  # noqa: E402


@pytest.mark.parametrize("name", SUPPORTED)
def test_model_shape_and_frozen_base(name):
    model = build_transfer_model(name, 7, weights=None)
    out = model.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)
    assert out.shape == (1, 7)
    trainable = {w.path.split("/")[0] for w in model.trainable_weights}
    assert all(not t.startswith(("vgg", "resnet", "mobilenet")) for t in trainable)
    assert len(model.trainable_weights) == 2  # uniquement la tête Dense (noyau + biais)


@pytest.mark.parametrize("name,fn", [
    ("mobilenetv2", keras.applications.mobilenet_v2.preprocess_input),
    ("resnet50", keras.applications.resnet50.preprocess_input),
    ("vgg16", keras.applications.vgg16.preprocess_input),
])
def test_preprocessing_matches_keras(name, fn):
    model = build_transfer_model(name, 3, weights=None)
    x = np.random.default_rng(1).uniform(0, 255, (2, *IMG_SIZE, 3)).astype("float32")
    mine = model.get_layer("preprocess")(x).numpy()
    np.testing.assert_allclose(mine, fn(x.copy()), atol=1e-3)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_transfer_model("alexnet", 3, weights=None)
