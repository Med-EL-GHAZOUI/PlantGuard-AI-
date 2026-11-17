"""Phase 13 : historique, suppression et statistiques (service factice, base SQLite en mémoire)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.prediction_service import Prediction


class FakeService:
    class_names = ["A___healthy", "Tomato___Early_blight", "Potato___Late_blight"]

    def __init__(self):
        self.queue = []

    def predict(self, image_bytes, top_k=3):
        name, conf = self.queue.pop(0) if self.queue else ("Tomato___Early_blight", 0.9)
        plant, disease = name.split("___")
        return Prediction(plant=plant, disease=disease.replace("_", " "), class_name=name, confidence=conf,
                          is_healthy=disease == "healthy", low_confidence=conf < 0.6,
                          top_k=[{"class_name": name, "confidence": conf}])


@pytest.fixture
def env(tmp_path):
    service = FakeService()
    settings = Settings(model_dir=tmp_path / "absent", database_url="sqlite:///:memory:",
                        upload_dir=tmp_path / "uploads")
    return TestClient(create_app(settings, service=service)), service, tmp_path / "uploads"


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (10, 120, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _predict(client):
    r = client.post("/api/predict", files={"file": ("../../evil name.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_predict_persists_analysis_with_uuid_filename(env):
    client, _, uploads = env
    body = _predict(client)
    assert body["id"] == 1 and body["image_url"] == "/api/analyses/1/image"
    files = list(uploads.iterdir())
    assert len(files) == 1 and files[0].suffix == ".png"
    assert "evil" not in files[0].name and len(files[0].stem) == 32  # nom généré, pas celui de l'utilisateur


def test_list_get_and_image(env):
    client, service, _ = env
    service.queue = [("A___healthy", 0.95), ("Potato___Late_blight", 0.7)]
    _predict(client); _predict(client)
    page = client.get("/api/analyses").json()
    assert page["total"] == 2 and [i["id"] for i in page["items"]] == [2, 1]  # plus récent d'abord
    one = client.get("/api/analyses/1").json()
    assert one["is_healthy"] is True and one["plant"] == "A"
    img = client.get(one["image_url"])
    assert img.status_code == 200 and img.content == _png()


def test_pagination_and_validation(env):
    client, _, _ = env
    for _ in range(3):
        _predict(client)
    assert len(client.get("/api/analyses?limit=2").json()["items"]) == 2
    assert [i["id"] for i in client.get("/api/analyses?limit=2&offset=2").json()["items"]] == [1]
    assert client.get("/api/analyses?limit=0").status_code == 422
    assert client.get("/api/analyses?limit=1000").status_code == 422


def test_unknown_analysis_404(env):
    client, _, _ = env
    assert client.get("/api/analyses/99").status_code == 404
    assert client.get("/api/analyses/99/image").status_code == 404
    assert client.delete("/api/analyses/99").status_code == 404


def test_delete_removes_row_and_file(env):
    client, _, uploads = env
    _predict(client)
    assert client.delete("/api/analyses/1").status_code == 204
    assert client.get("/api/analyses/1").status_code == 404
    assert list(uploads.iterdir()) == []


def test_stats(env):
    client, service, _ = env
    assert client.get("/api/stats").json()["total_analyses"] == 0
    service.queue = [("A___healthy", 0.9), ("Tomato___Early_blight", 0.9),
                     ("Tomato___Early_blight", 0.8), ("Potato___Late_blight", 0.7)]
    for _ in range(4):
        _predict(client)
    stats = client.get("/api/stats").json()
    assert (stats["total_analyses"], stats["healthy_count"], stats["diseased_count"]) == (4, 1, 3)
    assert stats["by_disease"][0] == {"class_name": "Tomato___Early_blight", "count": 2}
    assert stats["top_plants"][0] == {"plant": "Tomato", "count": 2}
    assert sum(d["count"] for d in stats["per_day"]) == 4


def test_failed_db_write_removes_image(tmp_path, monkeypatch):
    from backend.app.services import analysis_service
    settings = Settings(model_dir=tmp_path / "absent", database_url="sqlite:///:memory:",
                        upload_dir=tmp_path / "uploads")
    client = TestClient(create_app(settings, service=FakeService()), raise_server_exceptions=False)

    def boom(self):
        raise RuntimeError("db down")
    monkeypatch.setattr("sqlalchemy.orm.Session.commit", boom)
    r = client.post("/api/predict", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 500
    assert not (tmp_path / "uploads").exists() or list((tmp_path / "uploads").iterdir()) == []
