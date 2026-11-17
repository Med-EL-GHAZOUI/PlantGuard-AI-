"""Phase 7 : dégel partiel de la base, BatchNorm figée, sauvegarde/rechargement (weights=None)."""
import pytest

pytest.importorskip("tensorflow")

from tensorflow import keras  # noqa: E402

from ml.models.transfer import build_transfer_model, unfreeze_top_layers  # noqa: E402


def test_unfreeze_top_layers_keeps_batchnorm_frozen(tmp_path):
    model = build_transfer_model("mobilenetv2", 4, weights=None)
    head_only = len(model.trainable_weights)
    path = tmp_path / "m.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)  # même chemin que train_finetune

    n = unfreeze_top_layers(reloaded, 20)
    assert n > 0
    assert len(reloaded.trainable_weights) > head_only
    base = next(layer for layer in reloaded.layers if isinstance(layer, keras.Model))
    assert all(not l.trainable for l in base.layers if isinstance(l, keras.layers.BatchNormalization))
    assert all(not l.trainable for l in base.layers[:-20])


def test_invalid_unfreeze_count():
    model = build_transfer_model("mobilenetv2", 4, weights=None)
    with pytest.raises(ValueError):
        unfreeze_top_layers(model, 0)
