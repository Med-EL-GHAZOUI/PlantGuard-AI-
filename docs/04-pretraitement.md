# Phase 4 — Prétraitement et augmentation

## Principes
| Décision | Justification |
|---|---|
| Taille 224×224 (`ml/preprocessing/config.py`) | Entrée standard de VGG16, ResNet50 et MobileNetV2 |
| Images gardées en [0, 255] | La normalisation (`preprocess_input`) sera **dans le modèle** (Phase 5+) : entraînement et inférence appliquent ainsi la même transformation |
| Augmentation sur **train uniquement** | Validation et test restent intacts pour une mesure honnête |
| Pas de changement de teinte | La couleur (jaunissement, taches) est le signal des maladies |
| Flips H+V, rotation ±36°, zoom ±15 %, translation 10 %, luminosité/contraste ±15 % | Variations plausibles de cadrage et d'éclairage d'une photo de feuille |
| Poids de classes « balanced » | Compense le déséquilibre mesuré en Phase 3 (ratio 36,2) |

## Modules
| Fichier | Rôle | Dépend de TensorFlow |
|---|---|---|
| `preprocess.py` | `load_image` / `to_batch` : EXIF, RGB, transparence, redimensionnement (utilisé par l'API) | non |
| `class_info.py` | noms de classes, poids de classes, export JSON | non |
| `augmentation.py` | couches d'augmentation Keras | oui (import paresseux) |
| `data_pipeline.py` | `make_datasets` : train/val/test `tf.data` | oui (import paresseux) |

## Commande
```bash
python -m ml.preprocessing.class_info --train-dir ml/dataset/split/train --output-dir ml/artifacts
```
Produit `class_names.json` (indice → nom, lu par l'API) et `class_weights.json` pour `model.fit(class_weight=...)`.

## Résultats mesurés
38 classes ; poids de classes de 0,26 (classe la plus grande) à 9,44 (la plus petite).

## Limites
- Les parties TensorFlow (`augmentation.py`, `data_pipeline.py`) ne sont pas testées tant que TensorFlow n'est pas installé : leur test est ignoré (`importorskip`) et sera exécuté en Phase 5.
- Fond uniforme PlantVillage : l'augmentation ne simule pas les fonds réalistes (cf. Phase 3).
