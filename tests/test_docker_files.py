"""Phase 19 : garde-fous statiques sur les fichiers Docker (sans Docker) : cohérence avec le code et la sécurité."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_backend_image_runs_as_non_root_with_healthcheck():
    dockerfile = read("docker/backend.Dockerfile")
    assert re.search(r"^USER app$", dockerfile, re.M)
    assert "HEALTHCHECK" in dockerfile and "/api/health" in dockerfile
    assert "--factory" in dockerfile and "backend.app.main:create_app" in dockerfile


def test_backend_image_copies_only_what_inference_needs():
    dockerfile = read("docker/backend.Dockerfile")
    copied = re.findall(r"^COPY (\S+)", dockerfile, re.M)
    assert set(copied) == {"backend/requirements.txt", "backend", "ml/__init__.py", "ml/preprocessing"}
    ignore = read(".dockerignore")
    for heavy in ("ml/dataset", "ml/artifacts", "*.keras", ".venv", ".git", "tests"):
        assert heavy in ignore


def test_runtime_requirements_are_pinned_and_complete():
    lines = [l for l in read("backend/requirements.txt").splitlines() if l and not l.startswith("#")]
    assert all("==" in l for l in lines), "toutes les dépendances d'exécution doivent être figées"
    names = {l.split("==")[0] for l in lines}
    assert {"fastapi", "uvicorn", "python-multipart", "sqlalchemy", "tensorflow", "pillow", "numpy"} <= names
    assert "httpx" not in names and "pytest" not in names  # outils de test hors de l'image


def test_compose_hardening_and_no_published_backend_port():
    compose = read("docker-compose.yml")
    backend = compose.split("  frontend:")[0]
    assert "ports:" not in backend, "le backend ne doit pas être joignable depuis l'hôte"
    assert compose.count("read_only: true") == 2 and compose.count("cap_drop: [ALL]") == 2
    assert compose.count("no-new-privileges:true") == 2
    assert ":/models:ro" in compose and "plantguard-data:/data" in compose
    assert 'PLANTGUARD_RATE_LIMIT_PER_MINUTE: "0"' in compose  # la limite est faite par nginx


def test_every_compose_env_var_is_a_real_setting():
    compose = read("docker-compose.yml") + read("docker/backend.Dockerfile")
    used = set(re.findall(r"PLANTGUARD_[A-Z_]+", compose))
    defined = set(re.findall(r"PLANTGUARD_[A-Z_]+", read("backend/app/config.py")))
    assert used <= defined, f"variables inconnues du backend : {used - defined}"


def test_nginx_matches_dev_csp_and_limits_predict():
    from scripts.run_dev import content_security_policy

    nginx = read("docker/nginx.conf")
    # même politique que le serveur de dev, avec 'self' à la place de l'adresse de l'API (même origine)
    expected = content_security_policy("'self'").replace("img-src 'self' blob: 'self'", "img-src 'self' blob:")
    assert expected in nginx
    assert "limit_req zone=predict" in nginx and "limit_req_status 429" in nginx
    assert "server_tokens off" in nginx and "client_max_body_size 6m" in nginx
    assert "proxy_pass http://backend:8000" in nginx


def test_frontend_image_uses_same_origin_api_and_unprivileged_nginx():
    dockerfile = read("docker/frontend.Dockerfile")
    assert "nginx-unprivileged" in dockerfile
    assert "COPY docker/config.js /usr/share/nginx/html/config.js" in dockerfile
    assert 'window.PLANTGUARD_API = ""' in read("docker/config.js")


def test_frontend_treats_empty_api_address_as_same_origin():
    api_js = read("frontend/js/api.js")
    assert "window.PLANTGUARD_API ??" in api_js  # `||` transformerait "" en adresse de développement
