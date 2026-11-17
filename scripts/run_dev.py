"""Lance l'application complète en local : API (uvicorn) + frontend (serveur statique).

Le serveur du frontend génère lui-même `config.js` avec l'adresse de l'API choisie, et l'API reçoit
l'origine du frontend dans son CORS : les deux ports sont donc toujours cohérents.

Usage :
    python -m scripts.run_dev                       # API :8000, frontend :5500
    python -m scripts.run_dev --api-port 8100 --web-port 5600
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


class FrontendHandler(SimpleHTTPRequestHandler):
    """Sert `frontend/` ; `/config.js` est généré avec l'adresse de l'API (jamais lu sur disque)."""

    def __init__(self, *args, api_url: str, **kwargs):
        self.api_url = api_url
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 (nom imposé par http.server)
        if self.path.split("?")[0] == "/config.js":
            body = f'window.PLANTGUARD_API = "{self.api_url}";\n'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):  # sortie silencieuse
        pass


def make_frontend_server(api_url: str, port: int, directory: Path = FRONTEND_DIR,
                         host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = partial(FrontendHandler, directory=str(directory), api_url=api_url)
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-port", type=int, default=5500)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    for name, port in (("API", args.api_port), ("frontend", args.web_port)):
        if not port_is_free(port):
            raise SystemExit(f"Le port {port} ({name}) est déjà utilisé : choisis-en un autre "
                             f"avec --{'api' if name == 'API' else 'web'}-port.")

    api_url = f"http://127.0.0.1:{args.api_port}"
    web_url = f"http://127.0.0.1:{args.web_port}"
    os.environ["PLANTGUARD_CORS_ORIGINS"] = f"{web_url},http://localhost:{args.web_port}"
    if args.model_dir:
        os.environ["PLANTGUARD_MODEL_DIR"] = str(args.model_dir)

    server = make_frontend_server(api_url, args.web_port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Application : {web_url}   |   API : {api_url}/docs   (Ctrl+C pour arrêter)")

    import uvicorn
    try:
        uvicorn.run("backend.app.main:create_app", factory=True, host="127.0.0.1", port=args.api_port)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
