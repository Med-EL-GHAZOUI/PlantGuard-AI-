# Phase 15 — Intégration

## Lancer toute l'application
```bash
python -m scripts.run_dev                        # API :8000 + frontend :5500
python -m scripts.run_dev --api-port 8100 --web-port 5600   # si un port est occupé
```
`scripts/run_dev.py` vérifie que les ports sont libres (message clair sinon), génère `config.js` avec
l'adresse de l'API choisie et donne à l'API l'origine du frontend pour le CORS : les deux restent
toujours cohérents. Sans modèle exporté, l'API démarre en mode dégradé (bandeau d'avertissement dans l'interface).

## Tests d'intégration (`tests/test_integration.py`)
Vrai serveur HTTP (uvicorn en thread), vrai `PredictionService`, base SQLite **fichier**, requêtes `httpx` :
| Test | Vérifie |
|---|---|
| parcours utilisateur | santé → prédiction → détail → image identique → stats → suppression (ligne + fichier) |
| concurrence | 12 envois en parallèle (6 threads) : tous réussis, 12 identifiants distincts, 12 lignes |
| entrées invalides | un fichier refusé n'ajoute rien à l'historique |
| CORS | seule l'origine configurée est autorisée |
| serveur frontend | `config.js` généré, `index.html` servi ; détection de port occupé |

Suite complète : **68 tests passent**.

## Limites
- Le modèle utilisé en test est **non entraîné** : l'intégration est vérifiée, pas la qualité des prédictions.
- Le dernier maillon (modèle réellement entraîné → export → application) reste à faire après les entraînements (Phases 5-8).
- `run_dev` est un outil de développement ; la mise en production passera par Docker (Phase 19).
