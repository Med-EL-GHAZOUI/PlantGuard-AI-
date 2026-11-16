# Phase 0 — Analyse du projet PlantGuard AI

## 1. Contexte
Les maladies des plantes provoquent des pertes agricoles importantes. Leur identification repose
traditionnellement sur l'observation visuelle par un expert, ce qui est lent, coûteux et peu accessible.
La vision par ordinateur et les réseaux de neurones convolutifs (CNN) permettent d'automatiser
l'analyse d'images de feuilles.

## 2. Problématique
Comment concevoir un système basé sur les réseaux de neurones convolutifs capable d'identifier
automatiquement les maladies des plantes à partir d'images de feuilles avec une précision suffisante
pour constituer un outil expérimental d'aide à l'identification ?

## 3. Question scientifique
Parmi un CNN personnalisé et des architectures pré-entraînées (VGG16, ResNet50, MobileNetV2),
laquelle offre le meilleur compromis performance / taille / temps d'inférence pour classer les
maladies de feuilles, et dans quelle mesure se généralise-t-elle à des images hors du dataset ?

## 4. Objectifs généraux
- Construire un modèle de Deep Learning de classification plante + maladie.
- Intégrer le modèle dans une application Web complète (API + interface + base de données).

## 5. Objectifs spécifiques
1. Analyser et nettoyer le dataset, créer des splits train/validation/test sans fuite de données.
2. Implémenter un prétraitement et une augmentation cohérents biologiquement.
3. Entraîner un CNN baseline puis 3 modèles en Transfer Learning, avec fine-tuning.
4. Comparer les modèles par des métriques réelles (accuracy, precision, recall, F1, temps).
5. Ajouter l'explicabilité (Grad-CAM).
6. Exposer le modèle via une API REST FastAPI et stocker l'historique en SQLite.
7. Fournir une interface Web (upload, résultat, historique, dashboard).
8. Tester, sécuriser, conteneuriser (Docker) et documenter l'ensemble.

## 6. Fonctionnalités
F1 Upload d'image · F2 Analyse Deep Learning · F3 Résultat · F4 Base de connaissances maladies ·
F5 Historique · F6 Dashboard · F7 API REST.

## 7. Utilisateurs
- **Utilisateur final** (agriculteur, jardinier, étudiant) : envoie une photo, lit le résultat.
- **Évaluateur / enseignant** : consulte l'API, le dashboard et les résultats expérimentaux.
- **Développeur / mainteneur** : entraîne, déploie, fait évoluer le modèle.

## 8. Contraintes
- Projet académique, ressources de calcul limitées (CPU/GPU modeste).
- Stack imposée : HTML/CSS/JS vanilla, FastAPI, TensorFlow/Keras, SQLite.
- Outil expérimental : jamais un diagnostic professionnel.
- Formats acceptés : JPG, JPEG, PNG ; taille maximale limitée.

## 9. Hypothèses
- Le dataset (type PlantVillage) est représentatif des cas ciblés ; ses images sont souvent prises
  sur fond uniforme, ce qui peut **surestimer** la performance en conditions réelles (à discuter).
- Les images utilisateur sont des photos nettes d'une seule feuille.
- Un redimensionnement à 224×224 conserve l'information discriminante.

## 10. Résultats attendus
- Un modèle final choisi sur la base d'une comparaison expérimentale réelle (aucun chiffre inventé).
- Une application Web exécutable de bout en bout.
- Un dépôt GitHub propre, une documentation, un rapport et une présentation.

## Décisions techniques
| Décision | Justification |
|---|---|
| FastAPI | Léger, rapide, validation Pydantic, doc OpenAPI automatique |
| SQLite | Zéro configuration ; migration PostgreSQL prévue via SQLAlchemy |
| Transfer Learning | Peu de données/ressources : réutilise des features ImageNet |
| Frontend vanilla | Simple et compréhensible avant d'envisager un framework |
