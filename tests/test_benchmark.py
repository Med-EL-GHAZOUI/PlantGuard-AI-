"""Phase 18 : les outils de mesure eux-mêmes (aucune mesure de performance réelle n'est testée ici)."""
import pytest

from ml.evaluation.benchmark import summarize, synthetic_photo, time_calls, to_markdown
from ml.preprocessing.preprocess import load_image


def test_summarize_statistics():
    stats = summarize([10, 20, 30, 40, 100])
    assert stats["median_ms"] == 30 and stats["min_ms"] == 10 and stats["p95_ms"] == 100
    assert stats["mean_ms"] == 40


def test_summarize_single_sample():
    assert summarize([5.0])["p95_ms"] == 5.0


def test_time_calls_excludes_warmup_and_counts_runs():
    calls = []
    samples = time_calls(lambda: calls.append(1), runs=5, warmup=2)
    assert len(samples) == 5 and len(calls) == 7 and all(s >= 0 for s in samples)


def test_synthetic_photo_is_a_realistic_jpeg():
    data = synthetic_photo(1200, 900)
    assert data[:2] == b"\xff\xd8" and len(data) < 2_000_000  # JPEG, bien sous la limite d'upload
    assert load_image(data).shape == (224, 224, 3)


def test_markdown_flags_models_over_target():
    ok = {"model": "fast", "parameters": 1, "preprocess_small": {"median_ms": 1},
          "preprocess_large": {"median_ms": 2}, "model_only": {"median_ms": 3},
          "predict_small": {"median_ms": 4}, "predict_large": {"median_ms": 5, "p95_ms": 6}}
    slow = {**ok, "model": "slow", "predict_large": {"median_ms": 2500, "p95_ms": 3000}}
    lines = to_markdown([ok, slow]).splitlines()
    assert lines[2].endswith("oui |") and lines[3].endswith("NON |")
