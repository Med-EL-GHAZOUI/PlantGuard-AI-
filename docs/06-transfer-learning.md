# Phase 6 — Transfer Learning

## Modèles (`ml/models/transfer.py`)
VGG16, ResNet50, MobileNetV2 pré-entraînés sur ImageNet, sans leur tête (`include_top=False`), suivis de
GlobalAveragePooling → Dropout 0,3 → Dense softmax (38 classes).

| Choix | Justification |
|---|---|
| Base **gelée** en Phase 6 | Entraîne d'abord la tête seule : rapide, stable ; le fine-tuning est la Phase 7 |
| Base appelée avec `training=False` | Garde figées les statistiques BatchNorm (nécessaire pour un fine-tuning propre) |
| Prétraitement dans le modèle (conv 1×1 fixe) | Reproduit `preprocess_input` (MobileNetV2 : [-1,1] ; VGG16/ResNet50 : BGR − moyenne ImageNet), sérialisable sans `Lambda` ; vérifié identique à Keras par test |

## Entraînement
```bash
python -m ml.training.train_transfer --model mobilenetv2 --epochs 5
```
Sorties : `ml/artifacts/<modèle>/` (`best.keras`, `history.csv`, `summary.json`). Validation uniquement ;
le test reste réservé à la Phase 8. Les poids ImageNet sont téléchargés par Keras au premier lancement.

## Résultats
À compléter avec les valeurs réelles de chaque `summary.json` (non encore entraînés).
