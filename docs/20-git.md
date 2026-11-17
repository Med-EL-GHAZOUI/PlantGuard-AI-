# Phase 20 — Git et organisation du dépôt

## Conventions
| Sujet | Règle |
|---|---|
| Branche | `main` = état stable ; travail par branches `feat/…`, `fix/…`, `docs/…` puis Pull Request |
| Commits | Conventional Commits : `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:` ; un commit = un changement logique |
| Fins de ligne | `.gitattributes` : `* text=auto eol=lf` (LF dans le dépôt ; supprime les avertissements `LF will be replaced by CRLF`) ; binaires protégés |
| Secrets | jamais committés : `.env` ignoré, `.env.example` documente les variables (testé : aucune valeur sensible) |
| Fichiers lourds | `ml/dataset/raw/`, `split/`, `*.keras`, `*.h5`, `backend/uploads/*`, bases `*.db` : ignorés ; modèle à distribuer comme artefact de release |

## Intégration continue (`.github/workflows/ci.yml`)
| Job | Contenu |
|---|---|
| `tests` | Python 3.11, `pip install -r requirements-dev.txt`, `pytest --cov` |
| `dependency-audit` | `pip-audit -r backend/requirements.txt` |
| `docker-build` | `docker compose config -q` puis `docker compose build` |

Droits minimaux (`contents: read`), actions épinglées par version majeure, timeouts définis.

## Contrôles automatiques (`tests/test_repo_hygiene.py`)
Aucun fichier généré ou lourd suivi par Git (`.coverage`, `.keras`, `.db`, logs, uploads) · aucun fichier suivi
> 1 Mo · `.env.example` = exactement les variables lues par `config.py` · `.env` et artefacts ignorés ·
workflow CI valide, références existantes · README cite les points d'entrée réels et **toutes** les pages de docs.

## Ce que cette phase a corrigé
- `.coverage` et `ml/dataset/download.err` étaient **suivis par Git** (embarqués par un `git add -A`) : retirés de l'index et ignorés.
- Avertissements de fins de ligne à chaque commit : réglés par `.gitattributes`.
- README quasi vide : réécrit, avec un tableau d'état qui distingue ce qui est **fait**, **testé** et **mesuré** de ce qui reste à produire.

## Limites (honnêtes)
- **La CI n'a jamais tourné** : elle est validée syntaxiquement et ses commandes existent, mais elle n'a pas été exécutée sur GitHub. Les tests ont été lancés sous Windows ; un premier passage sous Linux pourrait révéler des différences.
- Le job `docker-build` est long (image TensorFlow de ~3 Go).
- **Aucune licence n'est choisie** : décision à prendre par l'auteur (MIT, Apache-2.0, ou tous droits réservés).
- Pas de protection de branche ni de revue obligatoire (réglages GitHub, hors dépôt).
- Pas de tags de version : à poser quand un modèle entraîné et évalué existera (par ex. `v0.1.0`).
