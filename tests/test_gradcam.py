"""Phase 9 : Grad-CAM sur modèles non entraînés et images SYNTHÉTIQUES (vérifie le câblage, pas la qualité)."""
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from ml.explainability.gradcam import compute_heatmap, explain, overlay_heatmap  # noqa: E402
from ml.models.baseline_cnn import build_baseline_cnn  # noqa: E402
from ml.models.transfer import build_transfer_model  # noqa: E402
from ml.preprocessing.config import IMG_SIZE  # noqa: E402

IMAGE = np.random.default_rng(0).uniform(0, 255, (*IMG_SIZE, 3)).astype("float32")


@pytest.mark.parametrize("build,expected", [
    (lambda: build_baseline_cnn(5), (14, 14)),                       # 224 / 2^4
    (lambda: build_transfer_model("mobilenetv2", 5, weights=None), (7, 7)),
])
def test_heatmap_shape_and_range(build, expected):
    heatmap, idx, prob = compute_heatmap(build(), IMAGE)
    assert heatmap.shape == expected
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert 0 <= idx < 5 and 0.0 <= prob <= 1.0


def test_explains_requested_class():
    model = build_baseline_cnn(5)
    _, idx, _ = compute_heatmap(model, IMAGE, class_index=3)
    assert idx == 3


def test_overlay_keeps_image_size():
    out = overlay_heatmap(IMAGE, np.random.default_rng(1).uniform(0, 1, (7, 7)))
    assert out.size == (IMG_SIZE[1], IMG_SIZE[0])


def test_explain_writes_file(tmp_path):
    model_path = tmp_path / "m.keras"
    build_baseline_cnn(3).save(model_path)
    Image.fromarray(IMAGE.astype("uint8")).save(tmp_path / "leaf.png")
    result = explain(model_path, tmp_path / "leaf.png", tmp_path / "out" / "cam.png", ["a", "b", "c"])
    assert (tmp_path / "out" / "cam.png").exists()
    assert result["class"] in {"a", "b", "c"}
