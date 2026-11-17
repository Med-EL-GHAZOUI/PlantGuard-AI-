"""Mesure la latence réelle d'inférence (cible du cahier des charges : < 2 s par image sur CPU).

La latence dépend de l'ARCHITECTURE, pas des poids : on peut donc la mesurer avant même l'entraînement
(`weights=None`). Pour chaque architecture on sépare :
  - décodage + prétraitement d'une photo (petite 256 px et grande 3000x2250 px, comme un téléphone) ;
  - appel du modèle seul (lot de 1) ;
  - `PredictionService.predict` complet (octets d'image -> résultat), la vraie chaîne de l'API.

Usage :
    python -m ml.evaluation.benchmark --runs 30
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ml.preprocessing.config import IMG_SIZE
from ml.preprocessing.preprocess import load_image


def summarize(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"median_ms": statistics.median(ordered), "p95_ms": p95, "min_ms": ordered[0],
            "mean_ms": statistics.fmean(ordered)}


def time_calls(fn, runs: int, warmup: int = 3) -> list[float]:
    """Durées (ms) de `runs` appels après `warmup` appels d'échauffement non comptés."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def synthetic_photo(width: int, height: int, fmt: str = "JPEG") -> bytes:
    """Image lisse et compressible (agrandissement d'un petit bruit), de taille réaliste pour une photo."""
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (height // 40, width // 40, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((width, height), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=88)
    return buf.getvalue()


def benchmark_architecture(name: str, runs: int) -> dict:
    from backend.app.services.prediction_service import PredictionService
    from ml.models.baseline_cnn import build_baseline_cnn
    from ml.models.transfer import build_transfer_model

    n_classes = 38
    model = build_baseline_cnn(n_classes) if name == "baseline" else build_transfer_model(
        name, n_classes, weights=None)
    service = PredictionService(model, [f"Plant___c{i}" for i in range(n_classes)])
    batch = np.zeros((1, *IMG_SIZE, 3), dtype="float32")
    small, large = synthetic_photo(256, 256), synthetic_photo(3000, 2250)

    return {
        "model": name,
        "parameters": int(model.count_params()),
        "preprocess_small": summarize(time_calls(lambda: load_image(small), runs)),
        "preprocess_large": summarize(time_calls(lambda: load_image(large), runs)),
        "model_only": summarize(time_calls(lambda: model(batch, training=False), runs)),
        "predict_small": summarize(time_calls(lambda: service.predict(small), runs)),
        "predict_large": summarize(time_calls(lambda: service.predict(large), runs)),
        "large_photo_kb": round(len(large) / 1024),
    }


def benchmark_api_concurrency(name: str, concurrencies=(1, 4, 8), requests_per_level: int = 24) -> list[dict]:
    """Débit et latence de bout en bout (HTTP, base SQLite, image stockée) avec N clients simultanés.

    Le service d'inférence est protégé par un verrou : on s'attend à un débit à peu près constant et à
    une latence qui croît avec le nombre de clients (les requêtes font la queue).
    """
    import socket
    import tempfile
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import uvicorn

    from backend.app.config import Settings
    from backend.app.main import create_app
    from backend.app.services.prediction_service import PredictionService
    from ml.models.transfer import build_transfer_model

    model = build_transfer_model(name, 38, weights=None)
    service = PredictionService(model, [f"Plant___c{i}" for i in range(38)])
    photo = synthetic_photo(1024, 768)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows : SQLite garde le fichier ouvert
        settings = Settings(database_url=f"sqlite:///{tmp}/bench.db", upload_dir=Path(tmp) / "up",
                            rate_limit_per_minute=0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(settings, service=service), host="127.0.0.1",
                                               port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            time.sleep(0.05)
        base = f"http://127.0.0.1:{port}"

        def one_request() -> float:
            start = time.perf_counter()
            response = httpx.post(f"{base}/api/predict", files={"file": ("p.jpg", photo, "image/jpeg")},
                                  timeout=120)
            response.raise_for_status()
            return (time.perf_counter() - start) * 1000

        results = []
        try:
            one_request()  # échauffement
            for clients in concurrencies:
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=clients) as pool:
                    latencies = list(pool.map(lambda _: one_request(), range(requests_per_level)))
                elapsed = time.perf_counter() - start
                results.append({"clients": clients, "requests": requests_per_level,
                                "throughput_rps": round(requests_per_level / elapsed, 2),
                                **{k: round(v, 1) for k, v in summarize(latencies).items()}})
        finally:
            server.should_exit = True
            thread.join(timeout=10)
    return results


def to_markdown(rows: list[dict], target_ms: float = 2000) -> str:
    head = ("| Modèle | Paramètres | Prétraitement 256 px | Prétraitement 3000 px | Modèle seul | "
            "predict() photo 256 px | predict() photo 3000 px | < 2 s ? |\n"
            "|---|---:|---:|---:|---:|---:|---:|:---:|")
    lines = []
    for r in rows:
        worst = r["predict_large"]["p95_ms"]
        lines.append(
            f"| {r['model']} | {r['parameters']:,} | {r['preprocess_small']['median_ms']:.0f} ms | "
            f"{r['preprocess_large']['median_ms']:.0f} ms | {r['model_only']['median_ms']:.0f} ms | "
            f"{r['predict_small']['median_ms']:.0f} ms | {r['predict_large']['median_ms']:.0f} ms "
            f"(p95 {worst:.0f}) | {'oui' if worst < target_ms else 'NON'} |")
    return "\n".join([head, *lines])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=["baseline", "mobilenetv2", "resnet50", "vgg16"])
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--api-model", default="mobilenetv2", help="architecture pour le test de concurrence HTTP ('' pour l'ignorer)")
    args = parser.parse_args()

    import os
    import tensorflow as tf

    rows = []
    for name in args.models:
        print(f"Mesure de {name}...", flush=True)
        rows.append(benchmark_architecture(name, args.runs))
    context = {"cpu_threads_available": os.cpu_count(), "platform": platform.platform(),
               "tensorflow": tf.__version__, "runs_per_measure": args.runs}
    concurrency = None
    if args.api_model:
        print(f"Concurrence HTTP ({args.api_model})...", flush=True)
        concurrency = {"model": args.api_model, "levels": benchmark_api_concurrency(args.api_model)}
        print(json.dumps(concurrency["levels"], indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps({"context": context, "results": rows, "api_concurrency": concurrency}, indent=2),
        encoding="utf-8")
    table = to_markdown(rows)
    (args.output_dir / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
