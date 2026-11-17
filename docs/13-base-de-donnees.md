# Phase 13 — Base de données (SQLite + SQLAlchemy)

## Modèle
Table `analyses` : `id`, `created_at` (UTC), `image_filename` (UUID généré, unique), `class_name`, `plant`,
`disease`, `confidence`, `is_healthy`, `low_confidence`. Index sur `created_at`, `class_name`, `plant`.
Les tables sont créées au démarrage (`create_all`). Migration PostgreSQL : changer `PLANTGUARD_DATABASE_URL`.

## Routes ajoutées
| Route | Rôle |
|---|---|
| `POST /api/predict` | enregistre maintenant l'analyse + l'image ; renvoie aussi `id` et `image_url` |
| `GET /api/analyses?limit=&offset=` | historique paginé (plus récent d'abord, `limit` 1-100) |
| `GET /api/analyses/{id}` · `/image` | détail et image de l'analyse |
| `DELETE /api/analyses/{id}` | supprime la ligne **et** le fichier image (204) |
| `GET /api/stats` | total, sains/malades, par maladie, plantes les plus analysées, évolution par jour |

## Choix de conception
- Images stockées dans `backend/uploads/` sous un nom UUID, jamais celui de l'utilisateur (aucun risque de traversée de chemin) ; seul le nom généré est en base.
- Si l'écriture en base échoue, l'image déjà écrite est retirée (testé).
- Toute la logique SQL est dans `services/analysis_service.py`, pas dans les routes.
- Variables d'environnement : `PLANTGUARD_DATABASE_URL` (défaut `sqlite:///backend/plantguard.db`), `PLANTGUARD_UPLOAD_DIR`.

## Limites
- Pas de migrations (Alembic) : un changement de schéma exige de recréer la base.
- Pas d'authentification : tout le monde voit l'historique de tous (conforme au cahier des charges initial).
- Dates en UTC sans fuseau (SQLite ne stocke pas le fuseau).
- Tests faits avec un service d'inférence factice et SQLite en mémoire.
