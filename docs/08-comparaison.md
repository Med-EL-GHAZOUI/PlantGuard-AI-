# Phase 8 — Comparaison des modèles

`ml/evaluation/compare_models.py` évalue chaque modèle sur le jeu de **test** (8 144 images, jamais
utilisé pour l'entraînement, l'early stopping ou le choix des hyperparamètres).

## Métriques
| Métrique | Pourquoi |
|---|---|
| Accuracy | Vue d'ensemble, mais biaisée par le déséquilibre des classes (ratio 36) |
| Precision / Recall / **F1 macro** | Chaque classe pèse pareil : critère principal de classement |
| Latence (ms/image) | Cible du cahier des charges : < 2 s sur CPU (mesurée après échauffement) |
| Taille (Mo), paramètres | Compromis performance / poids, pour le déploiement |
| Matrice de confusion normalisée | Montre quelles maladies sont confondues |

## Commande
```bash
python -m ml.evaluation.compare_models \
  --model baseline=ml/artifacts/baseline/best.keras \
  --model mobilenetv2=ml/artifacts/mobilenetv2_finetuned/best.keras \
  --model resnet50=ml/artifacts/resnet50_finetuned/best.keras \
  --model vgg16=ml/artifacts/vgg16_finetuned/best.keras
```
Sorties dans `docs/results/` : `comparison.md`, `comparison.json`, `confusion_<modèle>.png`.

## Résultats
À compléter avec la sortie réelle de la commande (aucun modèle n'a encore été évalué sur le test).
Le modèle final sera choisi sur F1 macro, latence et taille, puis discuté dans le rapport avec la
limite connue : fond uniforme PlantVillage, donc performance probablement surestimée en conditions réelles.
