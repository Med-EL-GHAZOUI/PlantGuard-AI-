"""Phase 20 : hygiène du dépôt (fichiers suivis, configuration d'environnement, CI)."""
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("dépôt Git indisponible")
    return out.splitlines()


def test_no_generated_or_heavy_files_are_tracked():
    forbidden = re.compile(r"(^|/)(\.coverage($|\.)|__pycache__/|\.pytest_cache/|\.venv/)|"
                           r"\.(pyc|keras|h5|db|log|err|pid)$|^\.env$")
    offenders = [f for f in tracked_files() if forbidden.search(f)]
    assert offenders == []
    uploads = [f for f in tracked_files() if f.startswith("backend/uploads/") and not f.endswith(".gitkeep")]
    assert uploads == []


def test_no_large_file_is_tracked():
    big = [f for f in tracked_files() if (ROOT / f).exists() and (ROOT / f).stat().st_size > 1_000_000]
    assert big == [], f"fichiers > 1 Mo suivis par Git : {big}"


def test_env_example_only_documents_real_settings():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    config = (ROOT / "backend/app/config.py").read_text(encoding="utf-8")
    documented = set(re.findall(r"^PLANTGUARD_[A-Z_]+", example, re.M))
    real = set(re.findall(r"PLANTGUARD_[A-Z_]+", config))
    assert documented == real, f"écart : {documented ^ real}"
    assert not re.search(r"(?i)(password|secret|token|api[_-]?key)\s*=\s*\S", example)


def test_env_files_are_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^\.env$", ignore, re.M)
    for pattern in ("*.keras", "ml/dataset/raw/", ".coverage"):
        assert pattern in ignore


def test_gitattributes_enforces_lf_and_protects_binaries():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes
    assert "*.keras binary" in attributes and "*.png binary" in attributes


def test_ci_workflow_is_valid_and_runs_the_real_commands():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert {"tests", "dependency-audit", "docker-build"} <= set(workflow["jobs"])
    commands = " ".join(str(step.get("run", "")) for job in workflow["jobs"].values() for step in job["steps"])
    assert "pip install -r requirements-dev.txt" in commands and "pytest" in commands
    assert "pip-audit -r backend/requirements.txt" in commands and "docker compose build" in commands
    # fichiers référencés par la CI : ils doivent exister
    for path in ("requirements-dev.txt", "backend/requirements.txt", "docker-compose.yml"):
        assert (ROOT / path).exists(), path
    assert workflow["permissions"] == {"contents": "read"}  # droits minimaux
    pinned = [step["uses"] for job in workflow["jobs"].values() for step in job["steps"] if "uses" in step]
    assert all(re.search(r"@v\d+$", u) for u in pinned)


def test_readme_documents_real_entry_points():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ("docker compose up --build", "python -m scripts.run_dev", "python -m pytest",
                   "experimental" if "experimental" in readme else "expérimental"):
        assert needle in readme, needle
    # chaque doc de phase est référencée depuis le README
    for doc in sorted((ROOT / "docs").glob("[0-9]*.md")):
        assert doc.name in readme, f"{doc.name} absent du README"
