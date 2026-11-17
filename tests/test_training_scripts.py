"""Phase 16 : les trois scripts d'entraînement tournent de bout en bout sur un mini-dataset SYNTHÉTIQUE.

1 époque, 3 classes, images aléatoires : on vérifie que le code s'exécute et produit ses fichiers,
pas que le modèle apprend quoi que ce soit. Les poids ImageNet ne sont pas téléchargés (weights=None).
"""
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from ml.models import transfer  # noqa: E402


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny") / "split"
    rng = np.random.default_rng(0)
    for split in ("train", "validation", "test"):
        for cls in ("A___healthy", "B___rust", "C___blight"):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(d / f"{i}.png")
    return root


@pytest.fixture
def offline_transfer(monkeypatch):
    original = transfer.build_transfer_model
    monkeypatch.setattr(transfer, "build_transfer_model",
                        lambda name, n, weights=None, freeze_base=True: original(name, n, weights=None,
                                                                                 freeze_base=freeze_base))


def _check_outputs(out_dir, summary):
    for name in ("best.keras", "history.csv", "summary.json"):
        assert (out_dir / name).exists(), name
    assert json.loads((out_dir / "summary.json").read_text()) == summary
    assert 0.0 <= summary["val_accuracy"] <= 1.0 and summary["val_loss"] > 0


def test_train_baseline_runs(split_dir, tmp_path):
    from ml.training.train_baseline import train

    summary = train(split_dir, tmp_path / "base", epochs=1, batch_size=4)
    _check_outputs(tmp_path / "base", summary)
    assert summary["model"] == "baseline_cnn" and summary["epochs_run"] == 1


def test_train_baseline_resume_continues_history_and_never_downgrades_best(split_dir, tmp_path):
    from ml.training.train_baseline import read_history, train

    out = tmp_path / "base"
    train(split_dir, out, epochs=1, batch_size=4)
    assert len(read_history(out)) == 1 and (out / "last.keras").exists()
    first_best = (out / "best.keras").stat().st_mtime_ns

    summary = train(split_dir, out, epochs=2, batch_size=4, resume=True)
    rows = read_history(out)
    assert [int(r["epoch"]) for r in rows] == [0, 1]  # la reprise ajoute l'époque 1, sans réécrire l'historique
    assert summary["epochs_run"] == 2
    assert (out / "best.keras").stat().st_mtime_ns >= first_best


def test_resume_without_checkpoint_or_when_already_done_fails_clearly(split_dir, tmp_path):
    from ml.training.train_baseline import train

    with pytest.raises(SystemExit):
        train(split_dir, tmp_path / "empty", epochs=2, batch_size=4, resume=True)
    out = tmp_path / "done"
    train(split_dir, out, epochs=1, batch_size=4)
    with pytest.raises(SystemExit):
        train(split_dir, out, epochs=1, batch_size=4, resume=True)


def test_train_transfer_then_finetune_run(split_dir, tmp_path, offline_transfer):
    from ml.training.train_finetune import finetune
    from ml.training.train_transfer import train

    stage1 = tmp_path / "mobilenetv2"
    head = train("mobilenetv2", split_dir, stage1, epochs=1, batch_size=4)
    _check_outputs(stage1, head)
    assert head["stage"] == "frozen_base"
    assert head["trainable_parameters"] < 5000  # uniquement la tête Dense (1280*3 + 3)

    tuned = finetune("mobilenetv2", split_dir, stage1, tmp_path / "ft", unfreeze=10,
                     epochs=1, batch_size=4, lr=1e-5)
    _check_outputs(tmp_path / "ft", tuned)
    assert tuned["stage"] == "finetuned" and tuned["unfrozen_layers"] > 0
    assert tuned["trainable_parameters"] > head["trainable_parameters"]  # le dégel a bien eu lieu


def test_finetune_requires_stage1_checkpoint(split_dir, tmp_path):
    from ml.training.train_finetune import finetune

    with pytest.raises(SystemExit):
        finetune("mobilenetv2", split_dir, tmp_path / "missing", tmp_path / "ft", 10, 1, 4, 1e-5)
