# Phase 9 — Explicabilité (Grad-CAM)

`ml/explainability/gradcam.py` produit une carte de chaleur montrant les zones de la feuille qui ont
le plus influencé la prédiction.

## Méthode
1. Passage avant jusqu'à la dernière carte convolutive (juste avant `GlobalAveragePooling2D`).
2. Gradient du score de la classe expliquée par rapport à cette carte.
3. Moyenne du gradient par canal = importance du canal ; carte pondérée, puis ReLU (influence positive seule).
4. Normalisation dans [0, 1], agrandissement et superposition (colormap `jet`).

Le même code sert au CNN baseline (carte 14×14) et aux modèles Transfer Learning (MobileNetV2 : 7×7) :
les couches sont appliquées en séquence car la base pré-entraînée est une couche imbriquée.

## Commande
```bash
python -m ml.explainability.gradcam --model ml/artifacts/mobilenetv2_finetuned/best.keras --image feuille.jpg
```

## Vérification et limites
- Les tests (modèles non entraînés, images synthétiques) vérifient le câblage : forme, plage [0, 1],
  choix de la classe, écriture du fichier. Ils ne prouvent **pas** que les cartes sont pertinentes.
- La pertinence (le modèle regarde-t-il la tache, ou le fond ?) devra être inspectée visuellement sur
  des exemples réels avec un modèle entraîné, et discutée dans le rapport. Sur PlantVillage (fond
  uniforme), un modèle peut s'appuyer sur des indices de fond : c'est précisément ce que Grad-CAM permet de détecter.
