"""Phase 8 : métriques et évaluation sur des données SYNTHÉTIQUES (aucun résultat réel)."""
import numpy as np
import pytest
from PIL import Image

from ml.evaluation.compare_models import compute_metrics, to_markdown


def test_metrics_perfect_and_imperfect():
    y = np.array([0, 1, 2, 2])
    assert compute_metrics(y, y)["f1_macro"] == 1.0
    m = compute_metrics(y, np.array([0, 1, 2, 0]))
    assert m["accuracy"] == 0.75 and m["f1_macro"] < 1.0


def test_macro_recall_penalises_ignored_small_class():
    y_true = np.array([0] * 9 + [1])
    y_pred = np.zeros(10, dtype=int)  # ignore totalement la classe 1
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.9 and m["recall_macro"] == 0.5


def test_markdown_sorted_by_f1():
    base = dict(accuracy=.5, precision_macro=.5, recall_macro=.5, latency_ms=10,
                size_mb=1.0, parameters=1000)
    md = to_markdown([{**base, "model": "low", "f1_macro": .2}, {**base, "model": "high", "f1_macro": .9}])
    assert md.index("high") < md.index("low")


def test_evaluate_model_end_to_end(tmp_path):
    pytest.importorskip("tensorflow")
    from ml.evaluation.compare_models import evaluate_model
    from ml.models.baseline_cnn import build_baseline_cnn
    from ml.preprocessing.data_pipeline import make_datasets

    rng = np.random.default_rng(0)
    for split in ("train", "validation", "test"):
        for cls in ("A___x", "B___y", "C___z"):
            d = tmp_path / "split" / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    _, _, test_ds, names = make_datasets(tmp_path / "split", batch_size=4, augment=False)
    model_path = tmp_path / "m.keras"
    build_baseline_cnn(3).save(model_path)

    row = evaluate_model("tiny", model_path, test_ds, names, tmp_path / "out")
    assert row["n_test_images"] == 12 and 0 <= row["accuracy"] <= 1
    assert row["latency_ms"] > 0 and row["size_mb"] > 0
    assert (tmp_path / "out" / "confusion_tiny.png").exists()
