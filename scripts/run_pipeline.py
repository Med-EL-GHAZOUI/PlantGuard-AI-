"""Enchaîne, UNE étape à la fois, tout ce qui reste à entraîner et à mesurer.

Pourquoi séquentiel : un seul calcul TensorFlow à la fois tient dans la mémoire du poste (un second a déjà
fait échouer l'entraînement). Chaque étape est ignorée si son fichier de résultat existe déjà, donc relancer
le script reprend là où il s'est arrêté.

Étapes :
  1. baseline          attend la fin de l'entraînement en cours (le relance avec --resume s'il est mort)
  2. mobilenetv2       transfert, base gelée                          -> ml/artifacts/mobilenetv2/summary.json
  3. mobilenetv2_ft    fine-tuning                                    -> ml/artifacts/mobilenetv2_finetuned/summary.json
  4. comparaison       jeu de TEST, les 3 modèles                     -> docs/results/comparison.json

Le script s'arrête après la comparaison : le choix du modèle final et son export restent une décision humaine.
Le jeu de test ne doit être utilisé qu'UNE fois pour choisir : ne pas relancer l'étape 4 pour « améliorer » le résultat.

Usage :
    python -m scripts.run_pipeline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = ROOT / "ml/artifacts/pipeline_status.json"
LOG_FILE = ROOT / "ml/artifacts/pipeline.log"


def log(message: str, path: Path | None = None) -> None:
    """Ajoute une ligne horodatée au journal (ouvert/fermé à chaque appel : rien n'est perdu si le processus est tué)."""
    path = path or LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} {message}\n")


@dataclass
class Stage:
    name: str
    command: list[str]
    done_file: Path
    retries: int = 1  # nombre de nouvelles tentatives après un premier échec
    env: dict = field(default_factory=dict)


def pid_alive(pid: int) -> bool:
    """Vrai si un processus de ce numéro existe (Windows : tasklist ; sinon signal 0)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_status(state: dict, path: Path = STATUS_FILE) -> None:
    """Écriture atomique : un lecteur ne voit jamais un fichier à moitié écrit."""
    state = {**state, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def run_stage(stage: Stage, status: dict, status_path: Path = STATUS_FILE, cwd: Path = ROOT) -> bool:
    """Exécute une étape (sauf si déjà faite). Retourne True si son fichier de résultat existe ensuite."""
    if stage.done_file.exists():
        status["stages"][stage.name] = {"state": "déjà fait"}
        write_status(status, status_path)
        return True
    for attempt in range(1, stage.retries + 2):
        status["stages"][stage.name] = {"state": "en cours", "attempt": attempt,
                                        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_status(status, status_path)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2", **stage.env}
        log(f"étape {stage.name} : tentative {attempt}")
        result = subprocess.run(stage.command, cwd=cwd, env=env)
        log(f"étape {stage.name} : code de retour {result.returncode}, résultat présent : {stage.done_file.exists()}")
        if result.returncode == 0 and stage.done_file.exists():
            status["stages"][stage.name] = {"state": "terminé", "attempt": attempt}
            write_status(status, status_path)
            return True
        status["stages"][stage.name] = {"state": "échec", "attempt": attempt, "returncode": result.returncode,
                                        "result_file_exists": stage.done_file.exists()}
        write_status(status, status_path)
    return False


def wait_for_baseline(stage: Stage, pid_file: Path, status: dict, status_path: Path = STATUS_FILE,
                      poll_seconds: float = 60.0, cwd: Path = ROOT, sleep=time.sleep) -> bool:
    """Attend `summary.json` du baseline. Si l'entraînement détaché est mort sans l'écrire, le reprend (--resume)."""
    relaunches = 0
    while not stage.done_file.exists():
        pid = int(pid_file.read_text().strip()) if pid_file.exists() and pid_file.read_text().strip().isdigit() else 0
        if pid_alive(pid):
            status["stages"][stage.name] = {"state": "en attente (entraînement en cours)", "pid": pid}
            write_status(status, status_path)
            log(f"attente du baseline (pid {pid} vivant)")
            sleep(poll_seconds)
            continue
        if relaunches > stage.retries:
            status["stages"][stage.name] = {"state": "échec", "reason": "entraînement mort, reprises épuisées"}
            write_status(status, status_path)
            return False
        relaunches += 1
        log(f"baseline mort sans summary.json : reprise n°{relaunches}")
        status["stages"][stage.name] = {"state": "reprise du baseline", "relaunch": relaunches}
        write_status(status, status_path)
        subprocess.run(stage.command, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"})
    status["stages"][stage.name] = {"state": "terminé"}
    write_status(status, status_path)
    return True


def default_stages(python: str = sys.executable) -> list[Stage]:
    m = [python, "-m"]
    art = ROOT / "ml/artifacts"
    return [
        Stage("baseline", m + ["ml.training.train_baseline", "--resume", "--epochs", "10"],
              art / "baseline/summary.json"),
        Stage("mobilenetv2", m + ["ml.training.train_transfer", "--model", "mobilenetv2", "--epochs", "5"],
              art / "mobilenetv2/summary.json"),
        Stage("mobilenetv2_ft", m + ["ml.training.train_finetune", "--model", "mobilenetv2", "--unfreeze", "30",
                                     "--epochs", "5"], art / "mobilenetv2_finetuned/summary.json"),
        Stage("comparaison", m + ["ml.evaluation.compare_models",
                                  "--model", f"baseline={art / 'baseline/best.keras'}",
                                  "--model", f"mobilenetv2={art / 'mobilenetv2/best.keras'}",
                                  "--model", f"mobilenetv2_finetuned={art / 'mobilenetv2_finetuned/best.keras'}"],
              ROOT / "docs/results/comparison.json"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-pid-file", type=Path, default=ROOT / "ml/artifacts/baseline_train.pid")
    args = parser.parse_args()

    import atexit
    import signal

    def on_signal(signum, _frame):
        log(f"signal reçu : {signum} : arrêt")
        raise SystemExit(128 + signum)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):  # selon le système
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)
    atexit.register(lambda: log("sortie du processus"))
    log("démarrage du pipeline")

    status: dict = {"stages": {}, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    stages = default_stages()
    ok = wait_for_baseline(stages[0], args.baseline_pid_file, status)
    for stage in stages[1:] if ok else []:
        ok = run_stage(stage, status)
        if not ok:
            break
    status["finished"] = ok
    write_status(status)
    print("Pipeline terminé." if ok else "Pipeline interrompu : voir ml/artifacts/pipeline_status.json")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
