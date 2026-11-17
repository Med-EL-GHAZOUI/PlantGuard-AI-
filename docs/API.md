# API REST — PlantGuard AI

Base : `http://localhost:8080` (Docker, nginx) ou `http://127.0.0.1:8000` (`scripts.run_dev`). Tous les chemins
commencent par `/api`. Documentation interactive : `/docs` (désactivée si `PLANTGUARD_ENABLE_DOCS=0`, cas par défaut sous Docker).
Les exemples de valeurs ci-dessous sont **illustratifs**.

## Routes
| Route | Rôle | Codes |
|---|---|---|
| `GET /api/health` | état de l'API et du modèle | 200 |
| `POST /api/predict` | analyser une image | 200, 400, 413, 415, 422, 429, 503 |
| `GET /api/analyses` | historique paginé | 200, 422 |
| `GET /api/analyses/{analysis_id}` | détail d'une analyse | 200, 404, 422 |
| `GET /api/analyses/{analysis_id}/image` | image de l'analyse | 200, 404, 422 |
| `DELETE /api/analyses/{analysis_id}` | supprimer l'analyse et son image | 204, 404, 422, 429 (hors Docker) |
| `GET /api/stats` | statistiques | 200 |
| `GET /api/diseases` | classes supportées et fiches | 200 |

## `GET /api/health`
```json
{ "status": "ok", "model_loaded": true, "model_name": "mobilenetv2" }
```
`status` vaut `"degraded"` et `model_loaded` `false` si le modèle n'a pas pu être chargé : l'API répond, mais `/predict` renvoie 503.

## `POST /api/predict`
Requête `multipart/form-data`, champ **`file`** (JPG, JPEG ou PNG, 5 Mo maximum, 25 millions de pixels maximum).
```bash
curl -F "file=@feuille.jpg;type=image/jpeg" http://localhost:8080/api/predict
```
Réponse 200 :
```json
{
  "id": 12,
  "image_url": "/api/analyses/12/image",
  "plant": "Tomato",
  "disease": "Early blight",
  "class_name": "Tomato___Early_blight",
  "confidence": 0.91,
  "is_healthy": false,
  "low_confidence": false,
  "top_k": [
    { "class_name": "Tomato___Early_blight", "confidence": 0.91 },
    { "class_name": "Tomato___Late_blight", "confidence": 0.05 },
    { "class_name": "Tomato___Target_Spot", "confidence": 0.02 }
  ],
  "disclaimer": "Outil expérimental d'aide à l'identification, pas un diagnostic professionnel."
}
```
| Champ | Sens |
|---|---|
| `confidence` | probabilité softmax de la classe prédite, entre 0 et 1. **Non calibrée** : ce n'est pas une garantie d'exactitude |
| `low_confidence` | `true` si `confidence` < seuil du modèle (`model_config.json`, 0,6 par défaut) : à présenter comme peu fiable |
| `is_healthy` | `true` si la maladie prédite est `healthy` |
| `top_k` | les 3 classes les plus probables, par ordre décroissant |

Chaque appel réussi est **enregistré** (ligne en base + image sous un nom UUID).

## `GET /api/analyses`
Paramètres : `limit` (1 à 100, défaut 20), `offset` (≥ 0, défaut 0). Plus récent d'abord.
```json
{ "total": 37,
  "items": [ { "id": 37, "created_at": "2026-11-17T02:25:10", "class_name": "Apple___healthy",
               "plant": "Apple", "disease": "healthy", "confidence": 0.88, "is_healthy": true,
               "low_confidence": false, "image_url": "/api/analyses/37/image" } ] }
```
`created_at` est en **UTC sans fuseau**. `GET /api/analyses/{analysis_id}` renvoie un seul de ces objets.

## `GET /api/analyses/{analysis_id}/image`
Renvoie le fichier d'origine (`image/jpeg` ou `image/png`) avec `X-Content-Type-Options: nosniff`.

## `DELETE /api/analyses/{analysis_id}`
204 sans corps ; supprime la ligne **et** le fichier image. **Irréversible.**

## `GET /api/stats`
```json
{ "total_analyses": 37, "healthy_count": 20, "diseased_count": 17,
  "by_disease": [ { "class_name": "Tomato___Early_blight", "count": 6 } ],
  "top_plants": [ { "plant": "Tomato", "count": 12 } ],
  "per_day":    [ { "date": "2026-11-17", "count": 37 } ] }
```
`by_disease` ne contient que les analyses non saines, triées par fréquence ; `top_plants` : 5 plantes au maximum.

## `GET /api/diseases`
Une entrée par classe supportée (38). `description` et `advice` valent `null` quand aucune fiche n'est rédigée (6 classes sur 38 en ont une).
```json
[ { "class_name": "Tomato___Early_blight", "plant": "Tomato", "disease": "Early blight",
    "is_healthy": false, "description": "…", "advice": "…" } ]
```
La liste est vide si ni le modèle ni `class_names.json` ne sont disponibles.

## Erreurs
| Code | Quand | Corps |
|---|---|---|
| 400 | fichier vide, illisible ou pas une image | `{"detail": "…"}` |
| 404 | analyse ou image inexistante | `{"detail": "Analyse introuvable."}` |
| 413 | fichier > 5 Mo, ou image > 25 millions de pixels | `{"detail": "…"}` |
| 415 | extension, type MIME ou format réel autre que JPEG / PNG | `{"detail": "…"}` |
| 422 | paramètre invalide (identifiant non entier, `limit` hors bornes, champ `file` absent) | format de validation FastAPI : `{"detail": [ {...} ]}` |
| 429 | trop de requêtes sur `predict` / `delete` | `{"detail": "…"}` + en-tête `Retry-After` (secondes) |
| 500 | erreur interne | `{"detail": "Erreur interne du serveur."}` (le détail reste dans les logs) |
| 503 | modèle non chargé | `{"detail": "Modèle indisponible : le service n'est pas prêt."}` |

Le 429 n'apparaît pas dans le schéma OpenAPI généré (la limitation est une dépendance) ; il est documenté ici.

## Limitation de débit
- **Docker** : nginx limite `POST /api/predict` à 30 requêtes/minute **par adresse IP** (rafale de 10) ; au-delà : 429. `DELETE` n'y est **pas** limité.
- **Local** (`run_dev`) : le backend limite `predict` et `delete` à 30/minute par client (`PLANTGUARD_RATE_LIMIT_PER_MINUTE`, 0 = désactivé).

## Sécurité côté API
Pas d'authentification : toute personne joignant l'API peut lire et **supprimer** l'historique de tous.
En-têtes sur toutes les réponses : `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
`Cache-Control: no-store`. CORS : seules les origines de `PLANTGUARD_CORS_ORIGINS` sont autorisées (inutile derrière nginx : même origine).
Voir [17-securite.md](17-securite.md).
