"""Protections transverses : en-têtes de sécurité et limitation de débit."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",          # le navigateur ne « devine » pas le type d'un fichier
    "X-Frame-Options": "DENY",                    # pas d'affichage dans une iframe (clickjacking)
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "cross-origin",  # les images doivent rester lisibles par le frontend (autre origine)
    "Cache-Control": "no-store",                  # réponses d'API non mises en cache
}


class SlidingWindowLimiter:
    """Au plus `limit` requêtes par client sur `window` secondes (en mémoire, par processus).

    `limit <= 0` désactive la limitation. L'identifiant du client est l'adresse TCP directe : on ne
    fait pas confiance à `X-Forwarded-For` (falsifiable). Derrière un reverse proxy, configurer le
    proxy pour limiter lui-même ou adapter cette clé.
    """

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self._clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Retourne None si la requête est autorisée, sinon le nombre de secondes à attendre."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            if len(self._hits) > 10_000:  # borne la mémoire : purge les clients inactifs
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return None


def rate_limit(request: Request) -> None:
    """Dépendance FastAPI : 429 + `Retry-After` quand le client dépasse sa limite."""
    client = request.client.host if request.client else "unknown"
    wait = request.app.state.rate_limiter.check(client)
    if wait is not None:
        raise HTTPException(429, "Trop de requêtes : réessayez dans un instant.",
                            headers={"Retry-After": str(int(wait) + 1)})
