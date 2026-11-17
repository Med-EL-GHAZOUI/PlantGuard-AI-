"""Orchestrateur d'entraînement : testé avec de fausses étapes (commandes Python minuscules, sans TensorFlow)."""
import json
import os
import sys
from pathlib import Path

import pytest

from scripts.run_pipeline import Stage, log, default_stages, pid_alive, run_stage, wait_for_baseline, write_status


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Les tests ne doivent jamais écrire dans le vrai journal du pipeline (ml/artifacts/pipeline.log)."""
    monkeypatch.setattr("scripts.run_pipeline.LOG_FILE", tmp_path / "test-pipeline.log")


def _py(code):
    return [sys.executable, "-c", code]


def _status():
    return {"stages": {}}


def test_stage_runs_and_records_success(tmp_path):
    done = tmp_path / "out.json"
    stage = Stage("s", _py(f"open(r'{done}', 'w').write('{{}}')"), done)
    status, sf = _status(), tmp_path / "status.json"
    assert run_stage(stage, status, sf, cwd=tmp_path) is True
    assert json.loads(sf.read_text(encoding="utf-8"))["stages"]["s"]["state"] == "terminé"


def test_finished_stage_is_skipped_not_rerun(tmp_path):
    done = tmp_path / "out.json"
    done.write_text("{}")
    marker = tmp_path / "ran"
    stage = Stage("s", _py(f"open(r'{marker}', 'w').write('x')"), done)
    assert run_stage(stage, _status(), tmp_path / "s.json", cwd=tmp_path) is True
    assert not marker.exists()  # la commande n'a pas été exécutée : relancer le pipeline reprend où il en était


def test_failed_stage_is_retried_then_succeeds(tmp_path):
    done, counter = tmp_path / "out.json", tmp_path / "count"
    code = (f"import pathlib,sys; c=pathlib.Path(r'{counter}'); n=int(c.read_text()) if c.exists() else 0; "
            f"c.write_text(str(n+1)); "
            f"sys.exit(1) if n == 0 else pathlib.Path(r'{done}').write_text('{{}}')")
    status = _status()
    assert run_stage(Stage("s", _py(code), done, retries=1), status, tmp_path / "s.json", cwd=tmp_path) is True
    assert counter.read_text() == "2" and status["stages"]["s"]["attempt"] == 2


def test_stage_gives_up_after_retries_and_reports_failure(tmp_path):
    done = tmp_path / "out.json"
    status = _status()
    assert run_stage(Stage("s", _py("import sys; sys.exit(3)"), done, retries=1), status, tmp_path / "s.json",
                     cwd=tmp_path) is False
    assert status["stages"]["s"]["state"] == "échec" and status["stages"]["s"]["returncode"] == 3


def test_exit_zero_without_result_file_counts_as_failure(tmp_path):
    """Une étape qui « réussit » sans produire son résultat ne doit pas être considérée comme faite."""
    done = tmp_path / "out.json"
    assert run_stage(Stage("s", _py("pass"), done, retries=0), _status(), tmp_path / "s.json", cwd=tmp_path) is False


def test_pid_alive():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False and pid_alive(-5) is False
    assert pid_alive(2_000_000_000) is False


def test_baseline_wait_returns_when_summary_appears(tmp_path):
    done, pid_file = tmp_path / "summary.json", tmp_path / "t.pid"
    pid_file.write_text(str(os.getpid()))  # « entraînement » vivant
    polls = []

    def fake_sleep(_):
        polls.append(1)
        if len(polls) == 2:
            done.write_text("{}")  # l'entraînement se termine pendant l'attente

    stage = Stage("baseline", _py("raise SystemExit(9)"), done)
    assert wait_for_baseline(stage, pid_file, _status(), tmp_path / "s.json", sleep=fake_sleep, cwd=tmp_path)
    assert len(polls) == 2


def test_baseline_dead_without_summary_is_resumed(tmp_path):
    done, pid_file = tmp_path / "summary.json", tmp_path / "t.pid"
    pid_file.write_text("2000000000")  # processus inexistant
    resume = Stage("baseline", _py(f"open(r'{done}', 'w').write('{{}}')"), done)  # la « reprise » réussit
    status = _status()
    assert wait_for_baseline(resume, pid_file, status, tmp_path / "s.json", sleep=lambda _: None, cwd=tmp_path)
    assert done.exists()


def test_baseline_dead_and_resume_keeps_failing_gives_up(tmp_path):
    done, pid_file = tmp_path / "summary.json", tmp_path / "t.pid"
    pid_file.write_text("2000000000")
    stage = Stage("baseline", _py("raise SystemExit(1)"), done, retries=1)
    status = _status()
    assert wait_for_baseline(stage, pid_file, status, tmp_path / "s.json", sleep=lambda _: None,
                             cwd=tmp_path) is False
    assert status["stages"]["baseline"]["state"] == "échec"  # pas de boucle infinie


def test_status_file_is_valid_json_with_timestamp(tmp_path):
    write_status({"stages": {"a": 1}}, tmp_path / "x.json")
    data = json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))
    assert data["stages"] == {"a": 1} and "updated_at" in data
    assert not list(tmp_path.glob("*.tmp"))


def test_default_pipeline_matches_real_scripts_and_options():
    """Chaque commande du pipeline doit viser un module existant avec des options qu'il connaît vraiment."""
    root = Path(__file__).resolve().parent.parent
    stages = default_stages()
    assert [s.name for s in stages] == ["baseline", "mobilenetv2", "mobilenetv2_ft", "comparaison"]
    for stage in stages:
        module = stage.command[stage.command.index("-m") + 1]
        source = (root / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        for arg in stage.command:
            if arg.startswith("--"):
                assert f'"{arg}"' in source, f"{stage.name} : option {arg} inconnue de {module}"
    # la comparaison lit exactement les fichiers que les étapes d'entraînement produisent
    compare = " ".join(stages[3].command).replace("\\", "/")
    assert "baseline/best.keras" in compare and "mobilenetv2_finetuned/best.keras" in compare
    assert stages[1].done_file.parent.name == "mobilenetv2" and stages[2].done_file.parent.name == "mobilenetv2_finetuned"


def test_log_appends_timestamped_lines(tmp_path):
    f = tmp_path / "p.log"
    log("premier", f)
    log("second", f)
    lines = f.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].endswith("premier") and f"pid={os.getpid()}" in lines[1]
    assert lines[0][:4].isdigit() and "T" in lines[0][:20]  # horodatage ISO
