# Phases 10 et 11 — Sauvegarde du modèle et service d'inférence

## Phase 10 — Export (`ml/export/export_model.py`)
Une fois le modèle retenu (Phase 8), on l'exporte avec sa configuration :

```bash
python -m ml.export.export_model --model ml/artifacts/mobilenetv2_finetuned/best.keras \
  --name mobilenetv2 --comparison docs/results/comparison.json
```
Produit `ml/artifacts/final/` :
| Fichier | Contenu |
|---|---|
| `model.keras` | Modèle complet, prétraitement inclus (entrée brute [0, 255]) |
| `class_names.json` | Ordre des classes (indice → nom) |
| `model_config.json` | Taille d'entrée, nb de classes, seuil de confiance (0,6 par défaut), SHA-256, version TensorFlow, métriques de test lues depuis `comparison.json` |

Garde-fous : l'export échoue si le nombre de classes du modèle ≠ celui de `class_names.json`.
`model.keras` est ignoré par Git (trop lourd) : à fournir comme artefact de release / volume Docker.

## Phase 11 — Inférence (`backend/app/services/prediction_service.py`)
`PredictionService.from_artifacts(dossier)` charge le modèle **une seule fois** (avec échauffement) ;
`predict(octets_image)` retourne `Prediction` : plante, maladie, classe, confiance, `is_healthy`,
`low_confidence` (confiance < seuil) et top-3.

- Vérifie l'empreinte SHA-256 : détecte un modèle modifié ou corrompu (`ModelLoadError`).
- Image invalide → `ValueError` (la couche API la traduira en erreur HTTP 400, Phase 12).
- Un verrou protège le modèle : FastAPI exécute les routes synchrones dans plusieurs threads.
- `low_confidence` permet à l'interface d'afficher un avertissement plutôt qu'un faux diagnostic sûr.

## Vérification
6 tests sur un petit modèle non entraîné et des images synthétiques : export, cohérence des classes,
prédiction, image invalide, fichier modifié, fichiers manquants. Les métriques réelles viendront de la Phase 8.
