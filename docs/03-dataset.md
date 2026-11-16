# Phase 3 — Dataset

## Dataset recommandé
**PlantVillage** (Kaggle) : photos de feuilles sur fond uniforme, un dossier par classe nommé `Plante___Maladie`.
Placer les dossiers de classes dans `ml/dataset/raw/` (ignoré par Git, trop volumineux).

> Attention : certaines versions Kaggle (« Augmented ») contiennent des images déjà transformées
> (rotations, flips) d'une même feuille. Le split par hash n'attrape que les doublons **exacts** ;
> préférer la version non augmentée et noter cette limite dans le rapport.

## Scripts
| Script | Rôle |
|---|---|
| `ml/preprocessing/dataset_utils.py` | lecture des classes, validation d'image, hash |
| `ml/preprocessing/analyze_dataset.py` | corrompues, comptage, déséquilibre, résolution, graphiques |
| `ml/preprocessing/split_dataset.py` | split 70/15/15 stratifié, doublons groupés, test anti-fuite |

## Commandes (depuis la racine du projet)
```bash
python -m ml.preprocessing.analyze_dataset --data-dir ml/dataset/raw --output-dir docs/figures
python -m ml.preprocessing.split_dataset --data-dir ml/dataset/raw --output-dir ml/dataset/split
python -m pytest tests -q
```

## Pourquoi train / validation / test
- **train** : le modèle apprend dessus.
- **validation** : choix des hyperparamètres, early stopping.
- **test** : évaluation finale, utilisé une seule fois, pour une mesure honnête.

## Résultats
Aucun chiffre ici tant que le dataset n'a pas été analysé : ils seront reportés après exécution réelle.
