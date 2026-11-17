"""Phase 15 : chaîne complète sur un VRAI serveur HTTP, un VRAI service d'inférence et une base SQLite fichier.

Le modèle est un petit réseau NON entraîné (poids aléatoires) et les images sont synthétiques : on
vérifie l'intégration des composants, pas la qualité des prédictions.
"""
import io
import json
import random
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

import uvicorn  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.main import create_app  # noqa: E402
from ml.export.export_model import export_model  # noqa: E402
from ml.models.baseline_cnn import build_baseline_cnn  # noqa: E402
from scripts.run_dev import make_frontend_server, port_is_free  # noqa: E402

NAMES = ["Apple___healthy", "Tomato___Early_blight", "Potato___Late_blight"]
ORIGIN = "http://127.0.0.1:5599"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _png(seed=0):
    buf = io.BytesIO()
    pixels = random.Random(seed).randbytes(64 * 64 * 3)  # déterministe : même graine = mêmes octets
    Image.frombytes("RGB", (64, 64), pixels).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    root = tmp_path_factory.mktemp("live")
    build_baseline_cnn(3).save(root / "src.keras")
    (root / "names.json").write_text(json.dumps(NAMES), encoding="utf-8")
    export_model(root / "src.keras", root / "names.json", root / "final", "integration-untrained")

    settings = Settings(model_dir=root / "final", database_url=f"sqlite:///{root / 'app.db'}",
                        upload_dir=root / "uploads", cors_origins=[ORIGIN])
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "le serveur de test n'a pas démarré"
    yield f"http://127.0.0.1:{port}", root
    server.should_exit = True
    thread.join(timeout=10)


def _upload(base, seed=0, name="leaf.png"):
    return httpx.post(f"{base}/api/predict", files={"file": (name, _png(seed), "image/png")}, timeout=60)


def test_full_user_journey(live):
    base, root = live
    assert httpx.get(f"{base}/api/health").json()["model_loaded"] is True

    r = _upload(base)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["class_name"] in NAMES and "disclaimer" in body

    # historique, image, statistiques cohérents avec ce qui vient d'être envoyé
    item = httpx.get(f"{base}/api/analyses/{body['id']}").json()
    assert item["class_name"] == body["class_name"]
    assert httpx.get(base + item["image_url"]).content == _png(0)
    assert httpx.get(f"{base}/api/stats").json()["total_analyses"] >= 1

    # suppression : ligne et fichier disparaissent
    uploads_before = len(list((root / "uploads").iterdir()))
    assert httpx.delete(f"{base}/api/analyses/{body['id']}").status_code == 204
    assert httpx.get(f"{base}/api/analyses/{body['id']}").status_code == 404
    assert len(list((root / "uploads").iterdir())) == uploads_before - 1


def test_concurrent_predictions_are_all_stored(live):
    base, _ = live
    total_before = httpx.get(f"{base}/api/analyses?limit=1").json()["total"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda i: _upload(base, seed=i), range(12)))
    assert all(r.status_code == 200 for r in responses)
    ids = {r.json()["id"] for r in responses}
    assert len(ids) == 12  # aucun identifiant en double
    assert httpx.get(f"{base}/api/analyses?limit=1").json()["total"] == total_before + 12


def test_bad_inputs_do_not_pollute_history(live):
    base, _ = live
    total = httpx.get(f"{base}/api/analyses?limit=1").json()["total"]
    bad = httpx.post(f"{base}/api/predict", files={"file": ("x.png", b"garbage", "image/png")})
    assert bad.status_code == 400
    assert httpx.get(f"{base}/api/analyses?limit=1").json()["total"] == total


def test_cors_only_for_configured_origin(live):
    base, _ = live
    ok = httpx.get(f"{base}/api/health", headers={"Origin": ORIGIN})
    bad = httpx.get(f"{base}/api/health", headers={"Origin": "http://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    assert "access-control-allow-origin" not in bad.headers


def test_frontend_server_generates_config_and_serves_index():
    port = _free_port()
    server = make_frontend_server("http://127.0.0.1:9999", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config = httpx.get(f"http://127.0.0.1:{port}/config.js")
        assert 'window.PLANTGUARD_API = "http://127.0.0.1:9999"' in config.text
        index = httpx.get(f"http://127.0.0.1:{port}/index.html")
        assert index.status_code == 200 and "PlantGuard AI" in index.text
    finally:
        server.shutdown()
        server.server_close()


def test_port_is_free_detects_busy_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert not port_is_free(s.getsockname()[1])
    assert port_is_free(_free_port())
