"""Phase 17 : en-têtes de sécurité, limitation de débit, docs masquables, chargement de modèle strict."""
import io
import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.security import SlidingWindowLimiter
from backend.app.services.prediction_service import Prediction
from scripts.run_dev import content_security_policy, make_frontend_server


class FakeService:
    class_names = ["A___healthy"]

    def predict(self, image_bytes, top_k=3):
        return Prediction("A", "healthy", "A___healthy", 0.9, True, False,
                          [{"class_name": "A___healthy", "confidence": 0.9}])


def _client(tmp_path, **kw):
    settings = Settings(model_dir=tmp_path / "absent", database_url="sqlite:///:memory:",
                        upload_dir=tmp_path / "uploads", **kw)
    return TestClient(create_app(settings, service=FakeService()))


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (5, 100, 20)).save(buf, format="PNG")
    return buf.getvalue()


def _predict(client):
    return client.post("/api/predict", files={"file": ("a.png", _png(), "image/png")})


# --- en-têtes -------------------------------------------------------------------------------

def test_security_headers_on_every_response(tmp_path):
    client = _client(tmp_path)
    for response in (client.get("/api/health"), client.get("/api/nope"), _predict(client)):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"


def test_uploaded_image_served_with_nosniff(tmp_path):
    client = _client(tmp_path)
    image = client.get(_predict(client).json()["image_url"])
    assert image.status_code == 200 and image.headers["x-content-type-options"] == "nosniff"
    assert image.headers["content-type"] == "image/png"


# --- limitation de débit --------------------------------------------------------------------

def test_predict_is_rate_limited_with_retry_after(tmp_path):
    client = _client(tmp_path, rate_limit_per_minute=3)
    assert [_predict(client).status_code for _ in range(3)] == [200, 200, 200]
    blocked = _predict(client)
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1
    assert client.get("/api/health").status_code == 200  # la limite ne touche pas les lectures


def test_delete_is_rate_limited_too(tmp_path):
    client = _client(tmp_path, rate_limit_per_minute=2)
    assert client.delete("/api/analyses/1").status_code == 404
    assert client.delete("/api/analyses/1").status_code == 404
    assert client.delete("/api/analyses/1").status_code == 429


def test_rate_limit_disabled_with_zero(tmp_path):
    client = _client(tmp_path, rate_limit_per_minute=0)
    assert all(_predict(client).status_code == 200 for _ in range(10))


def test_limiter_window_slides_and_isolates_clients():
    now = [0.0]
    limiter = SlidingWindowLimiter(2, window=60, clock=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(60)       # bloqué : attendre toute la fenêtre
    assert limiter.check("b") is None                    # un autre client n'est pas affecté
    now[0] = 61
    assert limiter.check("a") is None                    # la fenêtre a glissé


def test_limiter_is_thread_safe():
    limiter = SlidingWindowLimiter(50, window=60)
    allowed = []

    def hit():
        allowed.append(limiter.check("same") is None)

    threads = [threading.Thread(target=hit) for _ in range(200)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(allowed) == 50  # exactement la limite, jamais plus malgré la concurrence


# --- documentation de l'API -----------------------------------------------------------------

def test_docs_hidden_when_disabled(tmp_path):
    hidden, shown = _client(tmp_path, enable_docs=False), _client(tmp_path)
    assert hidden.get("/docs").status_code == 404 and hidden.get("/openapi.json").status_code == 404
    assert shown.get("/docs").status_code == 200


def test_enable_docs_env(monkeypatch):
    monkeypatch.setenv("PLANTGUARD_ENABLE_DOCS", "0")
    assert Settings.from_env().enable_docs is False
    monkeypatch.setenv("PLANTGUARD_ENABLE_DOCS", "1")
    assert Settings.from_env().enable_docs is True


# --- frontend -------------------------------------------------------------------------------

def test_csp_is_strict_and_scoped_to_api():
    csp = content_security_policy("http://127.0.0.1:8000")
    assert "script-src 'self'" in csp and "style-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "connect-src http://127.0.0.1:8000" in csp and "frame-ancestors 'none'" in csp


def test_frontend_served_with_csp():
    server = make_frontend_server("http://127.0.0.1:9", 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for path in ("/index.html", "/config.js", "/js/main.js"):
            response = httpx.get(f"http://127.0.0.1:{port}{path}")
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"], path
            assert response.headers["x-content-type-options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()


def test_frontend_has_no_inline_styles_or_scripts():
    """La CSP interdit l'inline : le code ne doit pas en dépendre (sinon l'interface casserait)."""
    from pathlib import Path
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert " style=" not in html and "onclick=" not in html
    inline_scripts = [s for s in html.split("<script")[1:] if 'src="' not in s.split(">")[0]]
    assert inline_scripts == []
    for js in Path("frontend/js").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert 'style: "' not in text and "innerHTML" not in text, js.name


# --- chargement du modèle -------------------------------------------------------------------

def test_model_with_arbitrary_code_layer_is_refused(tmp_path):
    """Un .keras contenant une couche Lambda (code arbitraire) doit être rejeté, même avec une empreinte valide."""
    pytest.importorskip("tensorflow")
    import json

    from tensorflow import keras

    from backend.app.services.prediction_service import ModelLoadError, PredictionService
    from ml.export.export_model import sha256_of

    inputs = keras.Input((224, 224, 3))
    x = keras.layers.Lambda(lambda t: t * 1.0)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.Model(inputs, keras.layers.Dense(2, activation="softmax")(x))
    final = tmp_path / "final"
    final.mkdir()
    model.save(final / "model.keras")
    (final / "class_names.json").write_text(json.dumps(["a", "b"]))
    # configuration forgée avec l'empreinte VALIDE du fichier : seule la couche Lambda doit le faire refuser
    (final / "model_config.json").write_text(json.dumps(
        {"sha256": sha256_of(final / "model.keras"), "confidence_threshold": 0.6}))
    with pytest.raises(ModelLoadError):
        PredictionService.from_artifacts(final)
