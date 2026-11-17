"""Phase 16 : tests de sécurité et de robustesse de l'API et de la configuration (sans TensorFlow)."""
import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.prediction_service import Prediction
from backend.app.utils.upload_validation import UploadValidationError, validate_upload


class FakeService:
    class_names = ["A___healthy"]

    def predict(self, image_bytes, top_k=3):
        return Prediction("A", "healthy", "A___healthy", 0.9, True, False,
                          [{"class_name": "A___healthy", "confidence": 0.9}])


def _client(tmp_path):
    settings = Settings(model_dir=tmp_path / "absent", database_url="sqlite:///:memory:",
                        upload_dir=tmp_path / "uploads")
    return TestClient(create_app(settings, service=FakeService()))


def _huge_png(width, height):
    """PNG valide d'une dizaine d'octets de données mais déclarant width x height pixels (bombe)."""
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    raw = zlib.compress(b"\x00" * (width * 3 + 1))  # une seule ligne : fichier tronqué mais en-tête valide
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def test_decompression_bomb_rejected_before_decoding():
    bomb = _huge_png(60000, 60000)  # 3,6 milliards de pixels
    assert len(bomb) < 1000
    with pytest.raises(UploadValidationError) as err:
        validate_upload("bomb.png", "image/png", bomb, max_bytes=5_000_000)
    assert err.value.status_code in (400, 413)


def test_large_image_below_pillow_limit_is_rejected_with_413():
    """9000x9000 = 81 M pixels : Pillow l'accepte sans même avertir, mais le décoder coûterait ~240 Mo."""
    with pytest.raises(UploadValidationError) as err:
        validate_upload("big.png", "image/png", _huge_png(9000, 9000), max_bytes=5_000_000)
    assert err.value.status_code == 413


def test_bomb_rejected_through_api(tmp_path):
    r = _client(tmp_path).post("/api/predict", files={"file": ("b.png", _huge_png(60000, 60000), "image/png")})
    assert r.status_code in (400, 413) and "detail" in r.json()


def test_truncated_png_rejected(tmp_path):
    buf = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (50, 50), (1, 2, 3)).save(buf, format="PNG")
    r = _client(tmp_path).post("/api/predict", files={"file": ("t.png", buf.getvalue()[:60], "image/png")})
    assert r.status_code == 400


@pytest.mark.parametrize("path", ["/api/analyses/abc", "/api/analyses/1;DROP TABLE analyses",
                                  "/api/analyses/1.5", "/api/analyses/abc/image"])
def test_non_integer_ids_rejected(tmp_path, path):
    assert _client(tmp_path).get(path).status_code == 422


def test_sql_injection_in_filename_is_harmless(tmp_path):
    client = _client(tmp_path)
    buf = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (20, 20)).save(buf, format="PNG")
    r = client.post("/api/predict", files={"file": ("x'; DROP TABLE analyses;--.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert client.get("/api/stats").json()["total_analyses"] == 1  # la table existe toujours


def test_wrong_http_methods_rejected(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/predict").status_code == 405
    assert client.post("/api/health").status_code == 405
    assert client.put("/api/analyses/1").status_code == 405


def test_unknown_route_404(tmp_path):
    assert _client(tmp_path).get("/api/secret").status_code == 404


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("PLANTGUARD_MAX_UPLOAD_MB", "2")
    monkeypatch.setenv("PLANTGUARD_CORS_ORIGINS", "http://a.test, http://b.test ,")
    monkeypatch.setenv("PLANTGUARD_MODEL_DIR", "somewhere")
    monkeypatch.setenv("PLANTGUARD_DATABASE_URL", "sqlite:///x.db")
    s = Settings.from_env()
    assert s.max_upload_bytes == 2 * 1024 * 1024
    assert s.cors_origins == ["http://a.test", "http://b.test"]
    assert str(s.model_dir) == "somewhere" and s.database_url == "sqlite:///x.db"


def test_config_defaults(monkeypatch):
    for key in ("PLANTGUARD_MAX_UPLOAD_MB", "PLANTGUARD_CORS_ORIGINS", "PLANTGUARD_MODEL_DIR",
                "PLANTGUARD_DATABASE_URL", "PLANTGUARD_UPLOAD_DIR"):
        monkeypatch.delenv(key, raising=False)
    s = Settings.from_env()
    assert s.max_upload_bytes == 5 * 1024 * 1024 and "http://localhost:5500" in s.cors_origins
