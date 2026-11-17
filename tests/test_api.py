"""Phase 12 : tests de l'API avec un service FACTICE (aucun vrai modèle, aucune donnée réelle)."""
import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.knowledge_base import load_knowledge
from backend.app.services.prediction_service import Prediction

NAMES = ["Apple___healthy", "Tomato___Early_blight"]


class FakeService:
    class_names = NAMES

    def predict(self, image_bytes, top_k=3):
        return Prediction(plant="Tomato", disease="Early blight", class_name=NAMES[1], confidence=0.91,
                          is_healthy=False, low_confidence=False,
                          top_k=[{"class_name": NAMES[1], "confidence": 0.91},
                                 {"class_name": NAMES[0], "confidence": 0.09}])


def _settings(tmp_path, **kw):
    return Settings(model_dir=tmp_path / "absent", **kw)


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(_settings(tmp_path), service=FakeService()))


def _img(fmt="PNG", size=(40, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, (20, 140, 30)).save(buf, format=fmt)
    return buf.getvalue()


def _post(client, data, name="leaf.png", mime="image/png"):
    return client.post("/api/predict", files={"file": (name, data, mime)})


def test_health_ok(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_success_includes_disclaimer(client):
    r = _post(client, _img())
    assert r.status_code == 200
    body = r.json()
    assert body["plant"] == "Tomato" and body["confidence"] == 0.91
    assert "pas un diagnostic professionnel" in body["disclaimer"]


def test_predict_accepts_jpeg(client):
    assert _post(client, _img("JPEG"), "a.jpg", "image/jpeg").status_code == 200


@pytest.mark.parametrize("name,mime,data,status", [
    ("leaf.gif", "image/gif", b"GIF89a", 415),                 # extension + MIME refusés
    ("leaf.png", "text/plain", None, 415),                      # MIME falsifié
    ("leaf.txt", "image/png", None, 415),                       # extension refusée
    ("leaf.png", "image/png", b"not an image", 400),            # faux contenu
    ("leaf.jpg", "image/jpeg", b"", 400),                       # vide
])
def test_predict_rejects_bad_uploads(client, name, mime, data, status):
    r = _post(client, _img() if data is None else data, name, mime)
    assert r.status_code == status
    assert "detail" in r.json()


def test_predict_rejects_other_image_format_renamed(client):
    assert _post(client, _img("GIF"), "leaf.png", "image/png").status_code == 415


def test_predict_rejects_oversized(tmp_path):
    client = TestClient(create_app(_settings(tmp_path, max_upload_bytes=1000), service=FakeService()))
    noise = io.BytesIO()
    Image.frombytes("RGB", (100, 100), os.urandom(100 * 100 * 3)).save(noise, format="PNG")  # incompressible
    assert _post(client, noise.getvalue()).status_code == 413


def test_predict_requires_file(client):
    assert client.post("/api/predict").status_code == 422


def test_degraded_mode_without_model(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    assert client.get("/api/health").json() == {"status": "degraded", "model_loaded": False,
                                                "model_name": None}
    r = _post(client, _img())
    assert r.status_code == 503


def test_internal_errors_do_not_leak(tmp_path):
    class Boom(FakeService):
        def predict(self, *a, **k):
            raise RuntimeError("secret path C:/internal")

    client = TestClient(create_app(_settings(tmp_path), service=Boom()), raise_server_exceptions=False)
    r = _post(client, _img())
    assert r.status_code == 500 and "secret" not in r.text


def test_diseases_lists_supported_classes(client):
    items = client.get("/api/diseases").json()
    assert [i["class_name"] for i in items] == NAMES
    assert items[0]["is_healthy"] is True
    assert items[1]["description"] is not None and items[0]["description"] is None


def test_cors_restricted(tmp_path):
    client = TestClient(create_app(_settings(tmp_path, cors_origins=["http://ok.local"]),
                                   service=FakeService()))
    ok = client.get("/api/health", headers={"Origin": "http://ok.local"})
    bad = client.get("/api/health", headers={"Origin": "http://evil.local"})
    assert ok.headers.get("access-control-allow-origin") == "http://ok.local"
    assert "access-control-allow-origin" not in bad.headers


def test_knowledge_keys_match_real_classes():
    names_file = Path("ml/artifacts/class_names.json")
    if not names_file.exists():
        pytest.skip("class_names.json absent")
    known = set(json.loads(names_file.read_text(encoding="utf-8")))
    assert set(load_knowledge()) <= known
