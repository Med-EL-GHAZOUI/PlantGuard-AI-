"""Phases 10-11 : export et service d'inférence avec un petit modèle NON entraîné et des images synthétiques."""
import io
import json

import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from backend.app.services.prediction_service import ModelLoadError, PredictionService  # noqa: E402
from ml.export.export_model import export_model  # noqa: E402
from ml.models.baseline_cnn import build_baseline_cnn  # noqa: E402

NAMES = ["Apple___healthy", "Tomato___Early_blight", "Potato___Late_blight"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    model_path = root / "src.keras"
    build_baseline_cnn(3).save(model_path)
    names_path = root / "class_names.json"
    names_path.write_text(json.dumps(NAMES), encoding="utf-8")
    export_model(model_path, names_path, root / "final", "tiny")
    return root / "final"


def _png(color=(30, 160, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), color).save(buf, format="PNG")
    return buf.getvalue()


def test_export_writes_config(exported):
    config = json.loads((exported / "model_config.json").read_text(encoding="utf-8"))
    assert config["n_classes"] == 3 and config["input_size"] == [224, 224]
    assert len(config["sha256"]) == 64 and config["test_metrics"] is None


def test_export_rejects_class_mismatch(tmp_path):
    model_path = tmp_path / "m.keras"
    build_baseline_cnn(3).save(model_path)
    names = tmp_path / "n.json"
    names.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    with pytest.raises(ValueError):
        export_model(model_path, names, tmp_path / "out", "x")


def test_service_predicts(exported):
    service = PredictionService.from_artifacts(exported)
    result = service.predict(_png())
    assert result.class_name in NAMES
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k) == 3
    assert abs(sum(t["confidence"] for t in result.top_k) - 1.0) < 1e-4  # 3 classes = toutes
    assert result.top_k[0]["confidence"] == result.confidence
    assert result.to_dict()["plant"] == result.plant


def test_service_rejects_invalid_image(exported):
    with pytest.raises(ValueError):
        PredictionService.from_artifacts(exported).predict(b"not an image")


def test_service_detects_tampered_model(exported, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(exported, copy)
    with open(copy / "model.keras", "ab") as f:
        f.write(b"x")
    with pytest.raises(ModelLoadError):
        PredictionService.from_artifacts(copy)


def test_service_reports_missing_files(tmp_path):
    with pytest.raises(ModelLoadError):
        PredictionService.from_artifacts(tmp_path)
