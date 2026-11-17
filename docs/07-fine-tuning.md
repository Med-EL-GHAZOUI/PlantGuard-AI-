# Phase 7 — Fine-tuning

Repart du modèle à base gelée (Phase 6, `ml/artifacts/<modèle>/best.keras`) et dégèle les N dernières
couches de la base (`unfreeze_top_layers`, défaut 30).

| Choix | Justification |
|---|---|
| Learning rate 1e-5 | Ajuste doucement sans détruire les features ImageNet |
| BatchNorm laissée figée | Évite d'instabiliser le fine-tuning avec de petits lots |
| Dégel des dernières couches seulement | Les premières couches (bords, textures) sont génériques |
| EarlyStopping sur `val_loss` | Arrête avant le sur-apprentissage |

```bash
python -m ml.training.train_finetune --model mobilenetv2 --unfreeze 30 --epochs 5
```
Sorties : `ml/artifacts/<modèle>_finetuned/` (`best.keras`, `history.csv`, `summary.json`). Validation
uniquement ; le test reste réservé à la Phase 8.

## Résultats
À compléter avec les `summary.json` réels (non encore entraînés). Le fine-tuning exige d'abord la Phase 6 entraînée.
