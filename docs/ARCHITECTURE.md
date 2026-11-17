# Architecture — PlantGuard AI

Les diagrammes UML de conception (Phase 2) sont dans [02-uml.md](02-uml.md) et `docs/uml/*.puml` ; ce document décrit
l'architecture **telle qu'elle est implémentée**.

## Vue d'ensemble
```text
                ┌──────────────── Docker : réseau interne ────────────────┐
navigateur ───► │ nginx :8080 ──────► backend uvicorn/FastAPI :8000        │
 (HTML/CSS/JS)  │ - fichiers statiques   │                                │
                │ - CSP, limite de débit │   ┌─ /models (lecture seule) : model.keras, class_names.json, model_config.json
                │ - proxy de /api/       └──►│
                └────────────────────────────┘   └─ /data (volume) : plantguard.db (SQLite) + uploads/<uuid>.jpg|png
```
Hors Docker : `python -m scripts.run_dev` lance l'API (:8000) et un petit serveur statique (:5500) qui génère `config.js` et pose la CSP.

## Couches du backend (`backend/app/`)
| Couche | Responsabilité | Ne fait **pas** |
|---|---|---|
| `main.py` | assemble l'application : CORS, en-têtes, limiteur, base, modèle, gestionnaire d'erreurs | aucune logique métier |
| `api/routes.py` | traduit HTTP ⇄ services (validation des paramètres, codes d'erreur) | pas de SQL, pas de TensorFlow |
| `services/prediction_service.py` | charge le modèle **une fois**, prédit à partir d'octets d'image, verrou | ne connaît pas HTTP |
| `services/analysis_service.py` | enregistrer / lister / supprimer / agréger les analyses (SQLAlchemy) | ne connaît pas HTTP |
| `services/knowledge_base.py` | fiches maladies (`data/diseases.json`) | |
| `utils/upload_validation.py` | contrôle d'un fichier envoyé (taille, extension, MIME, contenu, dimensions) | |
| `security.py` | en-têtes de sécurité, limiteur de débit à fenêtre glissante | |
| `database/`, `models/` | moteur SQLite et table `analyses` | |
| `schemas/` | contrats JSON (Pydantic) | |
| `config.py` | configuration par variables d'environnement | aucun secret dans le code |

Direction des dépendances : `routes → services → (ml.preprocessing, base)`. Le backend importe `ml.preprocessing`
(chargement d'image, noms de classes) pour que l'API et l'entraînement partagent **le même prétraitement** ; il n'importe
ni l'entraînement ni l'évaluation.

## Flux d'une analyse (`POST /api/predict`)
1. nginx : limite de débit par IP, taille ≤ 6 Mo, en-têtes de sécurité.
2. `rate_limit` (backend), puis lecture **bornée** du fichier (au plus `max + 1` octets en mémoire).
3. `validate_upload` : vide ? taille ? extension ? MIME ? ouverture réelle avec Pillow, format JPEG/PNG, dimensions ≤ 25 M pixels → sinon 400/413/415.
4. `PredictionService.predict` : `load_image` (EXIF, RGB, transparence → fond blanc, redimensionnement 224×224 en [0, 255]) → modèle (qui normalise lui-même) → softmax → top-3, plante / maladie, `low_confidence`.
5. `analysis_service.save_analysis` : image écrite sous un nom UUID, puis ligne SQL ; si la base échoue, l'image est retirée.
6. Réponse JSON avec `id`, `image_url` et l'avertissement « outil expérimental ».

## Modèle de données
Table unique `analyses` : `id`, `created_at` (UTC), `image_filename` (UUID, unique), `class_name`, `plant`, `disease`,
`confidence`, `is_healthy`, `low_confidence` ; index sur `created_at`, `class_name`, `plant`. Les tables sont créées au démarrage
(pas de migrations). Changer `PLANTGUARD_DATABASE_URL` permet de viser PostgreSQL (non testé).

## Décisions de conception et compromis
| Décision | Raison | Compromis |
|---|---|---|
| Normalisation **dans** le modèle (conv 1×1 fixe) | entraînement et inférence appliquent forcément la même transformation ; l'API envoie l'image brute | légèrement moins lisible qu'un `preprocess_input` externe |
| Prétraitement d'inférence avec Pillow | redimensionnement antialiasé, meilleur sur de grandes photos de téléphone | écart mesuré de 1,0/255 par pixel avec le redimensionnement TF de l'entraînement (à confirmer avec un modèle entraîné) |
| Modèle chargé une fois + verrou | éviter de recharger ; le modèle n'est pas garanti thread-safe | débit plafonné à ~4 req/s, file d'attente (mesuré, [18-performance.md](18-performance.md)) |
| Frontend et API sur la même origine (nginx) | pas de CORS à ouvrir, CSP simple | une brique de plus à opérer |
| Le modèle n'est pas dans l'image Docker | lourd, non versionné, remplaçable sans reconstruire | il faut l'exporter et le monter |
| `safe_mode=True` + empreinte SHA-256 | refuser le code arbitraire d'un `.keras` ; détecter la corruption | n'arrête pas un attaquant qui écrit modèle **et** configuration |
| SQLite | zéro configuration | un seul écrivain ; à migrer pour plusieurs instances |
| JS vanilla sans `innerHTML` | surface XSS minimale, CSP stricte possible | plus verbeux qu'un framework |

## Pipeline d'apprentissage (`ml/`)
`dataset` → `preprocessing` (analyse, split **sans fuite** par hachage, augmentation, poids de classes) → `models` /
`training` (baseline, transfert, fine-tuning) → `evaluation` (jeu de **test** utilisé une seule fois) → `export`
(modèle + configuration + empreinte) → chargé par le backend. Détails : [MODEL.md](MODEL.md).

## Stratégie de test
Plus de 130 tests (couverture mesurée de 89 % en Phase 16, avant les derniers ajouts) : unitaires, API (service factice), intégration (vrai serveur HTTP + vrai
service d'inférence + SQLite fichier), sécurité, scripts d'entraînement de bout en bout sur données synthétiques, fichiers Docker,
hygiène du dépôt, cohérence de la documentation. Limite : aucun test n'utilise un modèle **entraîné** ni de vraies photos.

## Points d'extension
- Nouveau modèle : l'exporter (`ml.export.export_model`), aucune modification du backend si la sortie est un softmax sur `class_names.json`.
- Nouvelle maladie : réentraîner avec la classe ajoutée ; ajouter sa fiche dans `backend/app/data/diseases.json` (clé = nom de classe).
- Authentification, quotas, stockage objet, plusieurs workers : non implémentés ; voir les limites de [17-securite.md](17-securite.md) et [18-performance.md](18-performance.md).
