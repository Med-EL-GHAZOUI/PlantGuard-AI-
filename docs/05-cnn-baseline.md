# Phase 5 — CNN baseline

## Architecture (`ml/models/baseline_cnn.py`)
Entrée 224×224×3 → `Rescaling(1/255)` (dans le modèle) → 4 × [Conv3×3 – BatchNorm – ReLU – MaxPool]
(32, 64, 128, 256 filtres) → GlobalAveragePooling → Dropout 0,4 → Dense softmax (38 classes).
GlobalAveragePooling plutôt que Flatten : moins de paramètres, moins de sur-apprentissage.

## Entraînement (`ml/training/train_baseline.py`)
Adam 1e-3, `class_weight` (Phase 4), augmentation sur train, EarlyStopping (patience 4, restaure les
meilleurs poids), ReduceLROnPlateau, ModelCheckpoint. Seule la **validation** est évaluée ; le test est
réservé à la Phase 8.

```bash
python -m ml.training.train_baseline --split-dir ml/dataset/split --epochs 10
```
Sorties dans `ml/artifacts/baseline/` (`best.keras`, `history.csv`, `summary.json`, non versionnées).

## Résultats
À compléter avec les valeurs réelles de `summary.json` une fois l'entraînement terminé
(mesuré : ~16 min par époque sur le CPU de la machine de développement).
