# Phase 12 — Backend FastAPI

## Lancement
```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:create_app --factory --reload
```
Documentation interactive : `http://127.0.0.1:8000/docs`. Sans modèle exporté (`ml/artifacts/final/`),
l'API démarre en **mode dégradé** : `/api/health` indique `degraded` et `/api/predict` répond 503.

## Endpoints de cette phase
| Route | Rôle |
|---|---|
| `GET /api/health` | État de l'API et du modèle |
| `POST /api/predict` | Image (`multipart`, champ `file`) → plante, maladie, confiance, top-3, avertissement |
| `GET /api/diseases` | Classes supportées + fiche (description, conseil) quand elle existe |

`GET/DELETE /api/analyses…` et `GET /api/stats` dépendent de la base de données : Phase 13.

## Sécurité (cahier des charges §10)
- Validation en 4 temps : taille (≤ 5 Mo, lecture bornée), extension, type MIME, **contenu réel** ouvert avec Pillow (format JPEG/PNG). Codes 400 / 413 / 415.
- Erreurs internes : détail dans les logs seulement ; le client reçoit « Erreur interne du serveur » (500).
- CORS limité aux origines de `PLANTGUARD_CORS_ORIGINS` (défaut : `localhost:5500`).
- Configuration par variables d'environnement : `PLANTGUARD_MODEL_DIR`, `PLANTGUARD_CORS_ORIGINS`, `PLANTGUARD_MAX_UPLOAD_MB`.

## Organisation
`main.py` (assemblage) · `api/routes.py` · `services/` (inférence, base de connaissances) · `schemas/` · `utils/upload_validation.py` · `config.py` · `data/diseases.json`.

## Limites
- La base de connaissances ne couvre que 6 des 38 classes (`description`/`advice` = `null` sinon) : à compléter et à faire relire par une source agronomique fiable avant tout usage réel.
- Les tests utilisent un service factice : la chaîne complète avec un vrai modèle entraîné n'a pas encore été exécutée.
- Le service d'inférence est protégé par un verrou : le débit est donc limité à une prédiction à la fois.
