"""Tests des scripts de la Phase 3 sur un mini-dataset SYNTHÉTIQUE (images aléatoires).

Ces images servent uniquement à vérifier que le code fonctionne ; elles ne représentent aucune donnée réelle.
"""
import numpy as np
from PIL import Image

from ml.preprocessing.analyze_dataset import analyze
from ml.preprocessing.dataset_utils import is_valid_image, parse_class_name
from ml.preprocessing.split_dataset import build_split, check_no_leakage

SPLITS = ("train", "validation", "test")


def _make_dataset(root, counts):
    rng = np.random.default_rng(0)
    for cls, n in counts.items():
        (root / cls).mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.jpg")


def test_parse_class_name():
    assert parse_class_name("Tomato___Early_blight") == ("Tomato", "Early blight")
    assert parse_class_name("Apple___healthy") == ("Apple", "healthy")


def test_corrupted_image_detected(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert not is_valid_image(bad)


def test_analyze_counts_and_corruption(tmp_path):
    _make_dataset(tmp_path / "raw", {"Tomato___Early_blight": 10, "Apple___healthy": 4})
    (tmp_path / "raw" / "Apple___healthy" / "broken.jpg").write_bytes(b"xx")
    summary = analyze(tmp_path / "raw", tmp_path / "fig")
    assert summary["n_images_valid"] == 14
    assert summary["n_images_corrupted"] == 1
    assert summary["n_classes"] == 2
    assert (tmp_path / "fig" / "class_distribution.png").exists()


def test_split_proportions_and_no_leakage(tmp_path):
    _make_dataset(tmp_path / "raw", {"A___x": 20, "B___y": 20})
    # duplique une image : elle ne doit pas finir dans deux splits différents
    src = tmp_path / "raw" / "A___x"
    (src / "copy.jpg").write_bytes((src / "0.jpg").read_bytes())
    report = build_split(tmp_path / "raw", tmp_path / "split")
    for split in SPLITS:
        assert sum(report[split].values()) > 0
    assert sum(sum(r.values()) for r in report.values()) == 41
    assert check_no_leakage(tmp_path / "split") == 0
