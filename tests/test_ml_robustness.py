"""Phase 16 : cas limites des scripts ML (données SYNTHÉTIQUES)."""
import json

import numpy as np
import pytest
from PIL import Image

from ml.preprocessing.class_info import main as class_info_main
from ml.preprocessing.split_dataset import build_split, check_no_leakage


def _make(root, cls, n, seed=0):
    rng = np.random.default_rng(seed)
    (root / cls).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(root / cls / f"{i}.png")


def test_leak_detector_actually_detects_a_leak(tmp_path):
    """Un détecteur qui ne détecte jamais rien serait inutile : on fabrique une vraie fuite."""
    for split in ("train", "validation", "test"):
        _make(tmp_path / split, "A___x", 2, seed=1)  # mêmes graines -> mêmes images dans chaque split
    assert check_no_leakage(tmp_path) == 2


def test_split_is_reproducible(tmp_path):
    _make(tmp_path / "raw", "A___x", 20)
    build_split(tmp_path / "raw", tmp_path / "s1")
    build_split(tmp_path / "raw", tmp_path / "s2")
    names = lambda d: sorted(p.name for p in d.rglob("*.png"))  # noqa: E731
    for split in ("train", "validation", "test"):
        assert names(tmp_path / "s1" / split) == names(tmp_path / "s2" / split)


def test_split_rejects_bad_proportions_and_non_empty_output(tmp_path):
    _make(tmp_path / "raw", "A___x", 10)
    with pytest.raises(ValueError):
        build_split(tmp_path / "raw", tmp_path / "o", train=0.9, val=0.2)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f.txt").write_text("x")
    with pytest.raises(FileExistsError):
        build_split(tmp_path / "raw", tmp_path / "full")


def test_split_tiny_class_does_not_crash(tmp_path):
    _make(tmp_path / "raw", "A___x", 2)
    report = build_split(tmp_path / "raw", tmp_path / "out")
    assert sum(r["A___x"] for r in report.values()) == 2


def test_split_ignores_corrupted_files(tmp_path):
    _make(tmp_path / "raw", "A___x", 9)
    (tmp_path / "raw" / "A___x" / "broken.png").write_bytes(b"junk")
    report = build_split(tmp_path / "raw", tmp_path / "out")
    assert sum(r["A___x"] for r in report.values()) == 9


def test_class_info_cli_writes_json(tmp_path, monkeypatch, capsys):
    _make(tmp_path / "train", "A___x", 3)
    _make(tmp_path / "train", "B___y", 1)
    monkeypatch.setattr("sys.argv", ["x", "--train-dir", str(tmp_path / "train"),
                                     "--output-dir", str(tmp_path / "art")])
    class_info_main()
    weights = json.loads((tmp_path / "art" / "class_weights.json").read_text())
    assert weights["1"] > weights["0"]  # la petite classe pèse plus
    assert json.loads((tmp_path / "art" / "class_names.json").read_text()) == ["A___x", "B___y"]
    assert "2 classes" in capsys.readouterr().out


def test_export_metrics_lookup_and_threshold(tmp_path):
    pytest.importorskip("tensorflow")
    from ml.export.export_model import export_model, load_metrics
    from ml.models.baseline_cnn import build_baseline_cnn

    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps([{"model": "m1", "f1_macro": 0.9}, {"model": "m2", "f1_macro": 0.5}]))
    assert load_metrics(comparison, "m2")["f1_macro"] == 0.5
    assert load_metrics(comparison, "absent") is None and load_metrics(None, "m1") is None

    build_baseline_cnn(2).save(tmp_path / "m.keras")
    (tmp_path / "n.json").write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError):
        export_model(tmp_path / "m.keras", tmp_path / "n.json", tmp_path / "o", "m1", None, 1.5)
    config = export_model(tmp_path / "m.keras", tmp_path / "n.json", tmp_path / "o", "m1", comparison, 0.7)
    assert config["confidence_threshold"] == 0.7 and config["test_metrics"]["f1_macro"] == 0.9


def test_gradcam_requires_global_pooling():
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    from ml.explainability.gradcam import compute_heatmap

    flat = keras.Sequential([keras.Input((8, 8, 3)), keras.layers.Flatten(), keras.layers.Dense(2)])
    with pytest.raises(ValueError):
        compute_heatmap(flat, np.zeros((8, 8, 3), dtype="float32"))
