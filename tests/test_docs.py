"""Phase 21 : la documentation ne doit pas dériver du code (routes, commandes, options, liens, variables)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FINAL_DOCS = ["ARCHITECTURE.md", "API.md", "MODEL.md", "INSTALLATION.md"]
NOT_PROJECT_MODULES = {"pytest", "venv", "pip", "http.server", "pip_audit", "uvicorn"}


def test_api_doc_matches_real_routes(tmp_path):
    from backend.app.config import Settings
    from backend.app.main import create_app

    app = create_app(Settings(model_dir=tmp_path / "none", database_url="sqlite:///:memory:",
                              upload_dir=tmp_path / "up"))
    real = {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}
    text = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(GET|POST|DELETE|PUT|PATCH) (/api/[^`\s]*)`", text))
    assert real == documented, f"routes non documentées : {real - documented} ; routes inexistantes : {documented - real}"


def test_documented_status_codes_cover_openapi(tmp_path):
    from backend.app.config import Settings
    from backend.app.main import create_app

    app = create_app(Settings(model_dir=tmp_path / "none", database_url="sqlite:///:memory:",
                              upload_dir=tmp_path / "up"))
    text = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    table = {}
    for line in text.splitlines():
        m = re.match(r"\| `(GET|POST|DELETE) (/api/\S+)` \|.*\| ([\d, ()a-zA-Zè ]+) \|$", line)
        if m:
            table[(m.group(1), m.group(2))] = set(re.findall(r"\d{3}", m.group(3)))
    for path, ops in app.openapi()["paths"].items():
        for method, op in ops.items():
            assert set(op["responses"]) <= table[(method.upper(), path)], (method, path)


def _commands(text):
    """Lignes de commande des blocs de code, continuations avec antislash jointes."""
    joined = re.sub(r"\\n\s*", " ", text)
    return [m for m in re.findall(r"python -m ([\w.]+)([^\n`]*)", joined)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_documented_command_exists_with_real_options(doc):
    for module, rest in _commands(doc.read_text(encoding="utf-8")):
        if module in NOT_PROJECT_MODULES:
            continue
        source = ROOT / (module.replace(".", "/") + ".py")
        assert source.exists(), f"{doc.name} : module inexistant `python -m {module}`"
        code = source.read_text(encoding="utf-8")
        for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest):
            assert f'"{flag}"' in code, f"{doc.name} : option {flag} inconnue de {module}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+://", target):
            continue
        assert (doc.parent / target).resolve().exists(), f"{doc.name} : lien cassé {target}"


def test_documented_environment_variables_are_real():
    real = set(re.findall(r"PLANTGUARD_[A-Z_]+", (ROOT / "backend/app/config.py").read_text(encoding="utf-8")))
    for doc in DOCS:
        mentioned = set(re.findall(r"PLANTGUARD_[A-Z_]+", doc.read_text(encoding="utf-8")))
        assert mentioned <= real, f"{doc.name} : variables inconnues {mentioned - real}"
    installation = (ROOT / "docs/INSTALLATION.md").read_text(encoding="utf-8")
    assert real <= set(re.findall(r"PLANTGUARD_[A-Z_]+", installation)), "INSTALLATION.md doit toutes les citer"


def test_final_docs_exist_and_are_referenced_from_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in FINAL_DOCS:
        assert (ROOT / "docs" / name).exists() and f"docs/{name}" in readme, name


def test_model_doc_states_when_results_are_missing():
    """Tant qu'aucune comparaison n'est faite, MODEL.md doit le dire et ne contenir aucun résultat de test."""
    text = (ROOT / "docs/MODEL.md").read_text(encoding="utf-8")
    if not (ROOT / "docs/results/comparison.json").exists():
        assert "aucune comparaison" in text.lower() or "À produire" in text
        assert "jeu de test" in text and "ne sont **pas** des performances de test" in text
    else:
        pytest.skip("comparaison disponible : MODEL.md doit être mis à jour avec ses chiffres")
