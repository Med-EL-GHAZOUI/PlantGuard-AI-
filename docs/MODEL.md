# Modèle — PlantGuard AI

> **État : les modèles ne sont pas encore évalués.** Seul le CNN baseline est en cours d'entraînement (interrompu à 3 reprises
> par un épuisement de mémoire du poste, voir plus bas) ; aucune
> comparaison sur le jeu de test n'a été faite. Ce document décrit la méthode et ne contient que des chiffres **mesurés**,
> clairement étiquetés. Les sections « Résultats » seront complétées avec la sortie réelle de `compare_models`.

## Tâche
Classification d'une image de feuille parmi **38 classes** « Plante___Maladie » (14 plantes, dont des classes `healthy`).
Un seul modèle prédit plante et maladie ensemble (la plante est déduite du nom de classe).

## Données (mesurées en Phase 3)
Dataset PlantVillage (variante couleur, non augmentée) : **54 305** images valides, 0 corrompue, 256×256 RGB, 38 classes.
Déséquilibre : de 152 (`Potato___healthy`) à 5 507 images (`Orange___Haunglongbing_(Citrus_greening)`), ratio **36,2**.
21 doublons exacts. Découpage 70/15/15, stratifié par classe, graine 42, doublons regroupés dans le même sous-ensemble :

| Sous-ensemble | Images | Usage |
|---|---:|---|
| train | 38 014 | apprentissage |
| validation | 8 147 | early stopping, choix des hyperparamètres |
| test | 8 144 | **mesure finale uniquement**, via `compare_models` |

Fuites détectées entre sous-ensembles (par hachage) : 0. Limite : le hachage ne repère que les doublons **exacts** ;
des photos d'une même feuille sous des angles différents peuvent exister dans plusieurs sous-ensembles et gonfler la performance.

## Prétraitement et augmentation
Redimensionnement 224×224, valeurs gardées en [0, 255] ; la normalisation propre à chaque architecture est **dans le modèle**.
Augmentation (train uniquement) : flips horizontal et vertical, rotation ±36°, zoom ±15 %, translation 10 %, luminosité et
contraste ±15 %. **Pas de changement de teinte** : la couleur est le signal des maladies. Déséquilibre : poids de classes
« balanced » (de 0,26 à 9,44) ; l'évaluation utilise le **F1 macro**, pas la seule accuracy.

## Architectures
| Modèle | Paramètres totaux (mesurés, 38 classes) | Entraînés à l'étape « base gelée » (calculé) |
|---|---:|---:|
| CNN baseline (4 × Conv-BN-ReLU-MaxPool, GAP, Dropout 0,4) | 399 622 | tous |
| MobileNetV2 (ImageNet) | 2 306 674 | 48 678 (tête Dense) |
| ResNet50 (ImageNet) | 23 665 586 | 77 862 |
| VGG16 (ImageNet) | 14 734 194 | 19 494 |

Transfert : base pré-entraînée `include_top=False` → GlobalAveragePooling → Dropout 0,3 → Dense softmax. Base appelée avec
`training=False` (statistiques BatchNorm figées).

## Protocole d'entraînement
1. **Baseline** : Adam 1e-3, `class_weight`, EarlyStopping sur `val_loss` (patience 4, restaure les meilleurs poids), ReduceLROnPlateau, 10 époques maximum. `--resume` reprend après interruption.
2. **Transfert, base gelée** : Adam 1e-3, tête seule, 5 époques par défaut.
3. **Fine-tuning** : dégel des 30 dernières couches de la base (BatchNorm figée), Adam **1e-5**, repart du modèle de l'étape 2.
4. Graine 42 ; la validation sert à choisir, **jamais le test**.

## Protocole d'évaluation (Phase 8)
`python -m ml.evaluation.compare_models --model nom=chemin.keras ...` sur le jeu de **test** : accuracy, precision / recall / F1
macro, F1 pondéré, matrice de confusion normalisée, latence par image, taille du fichier, paramètres. Classement sur le **F1 macro**.
Le modèle final sera choisi sur F1 macro, latence et taille, puis exporté (`ml.export.export_model`).

## Contrat d'inférence
Entrée : image brute décodée par `ml.preprocessing.preprocess.load_image` → `(1, 224, 224, 3)` float32 en [0, 255].
Sortie : 38 probabilités (softmax) ordonnées comme `class_names.json` (ordre alphabétique des noms de classe).
`low_confidence = confiance < seuil` (0,6 par défaut, `model_config.json`).

## Explicabilité
Grad-CAM (`ml.explainability.gradcam`) : carte de chaleur de la dernière carte convolutive. Testée sur le câblage uniquement ;
la pertinence (le modèle regarde-t-il la tache ou le fond ?) reste **à inspecter visuellement** sur un modèle entraîné.

## Résultats

### CNN baseline — validation uniquement (relevé du 17/11/2026 à 02:53, entraînement en cours)
| Époque | Accuracy entraînement (augmentée) | Accuracy validation | Perte validation |
|---:|---:|---:|---:|
| 1 | 0,471 | 0,407 | 2,317 |
| 2 | 0,672 | 0,585 | 1,436 |
| 3 | 0,760 | 0,470 | 2,214 |
| 4 | 0,797 | 0,628 | 1,639 |
| 5 | 0,842 | 0,754 | 0,775 |

**Incidents d'entraînement (réels).** Le processus a disparu à ~01:40 (époque 5), puis à ~02:56 (époque 6), puis a échoué au pas 4 avec
`could not create a memory object` (oneDNN) ; Windows a ensuite redémarré à 03:06. Le détecteur d'épuisement de ressources de Windows
indique, aux heures concernées, une insuffisance de mémoire virtuelle dont le premier consommateur est la machine virtuelle WSL de Docker
Desktop (~10,7 Go), devant l'entraînement (4 à 6 Go) et une base MySQL. Cause **probable**, non prouvée pour les deux premiers arrêts.
La reprise (`--resume`) repart de `last.keras` : au plus une époque perdue.

Lecture prudente : la validation est irrégulière (époques 3 et 4) et l'entraînement n'est pas terminé. L'accuracy
d'entraînement est mesurée sur des images **augmentées**. Ces chiffres ne sont **pas** des performances de test. La valeur courante est dans
`ml/artifacts/baseline/history.csv`.

### Comparaison finale sur le jeu de test
**À produire** : aucun modèle de transfert n'est entraîné, aucune évaluation sur le test n'a été exécutée.

### Latence d'inférence (architecture, poids aléatoires, CPU 20 threads, machine non dédiée)
Médiane de `predict()` (octets d'une photo 3000×2250 → résultat) : baseline 105 ms, MobileNetV2 250 ms, ResNet50 365 ms, VGG16 212 ms ;
toutes sous la cible de 2 s ([18-performance.md](18-performance.md)).

## Limites et risques
- **Généralisation non mesurée** : PlantVillage = feuilles isolées sur fond uniforme. Une performance élevée sur le test n'implique pas une performance élevée sur des photos prises au champ ; aucun jeu de test « hors dataset » n'existe dans ce projet.
- **Confiance non calibrée** : le seuil de 0,6 est un choix **arbitraire**, non justifié par une mesure ; une probabilité softmax élevée ne garantit pas l'exactitude, et le modèle n'a pas de classe « autre / pas une feuille » (une image quelconque reçoit quand même une prédiction).
- Doublons quasi identiques (angles, recadrages) possibles entre sous-ensembles ; classes rares (152 images) : métriques par classe moins fiables.
- Écart de redimensionnement TensorFlow (entraînement) / Pillow (inférence) : 1,0/255 en moyenne par pixel sur 20 images ; son effet sur les prédictions reste à mesurer.
- Résultats d'un seul entraînement par modèle (une graine) : pas d'intervalle de confiance.
- Outil expérimental : ne jamais l'utiliser pour une décision agronomique sans avis d'un professionnel.
