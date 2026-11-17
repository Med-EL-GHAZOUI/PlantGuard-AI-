# Rapport — PlantGuard AI : identification de maladies de feuilles par réseaux de neurones convolutifs

> **BROUILLON.** Les sections marquées **[À COMPLÉTER — Phase 8]** attendent les résultats réels de la comparaison sur le
> jeu de test, qui n'ont pas encore été produits. Aucun résultat n'est inventé ici : ce qui est chiffré a été mesuré et
> renvoie à sa source. Les références bibliographiques sont à **vérifier** (titre, pages) avant remise.

## Résumé
*[À COMPLÉTER — Phase 8 : une fois le modèle retenu, résumer en 5 phrases : problème, méthode, meilleur modèle et ses métriques
de test, limite principale.]*

## 1. Introduction

### 1.1 Contexte
Les maladies foliaires provoquent des pertes agricoles importantes. Leur identification repose classiquement sur l'observation
par un expert, lente, coûteuse et peu accessible. La vision par ordinateur et les réseaux de neurones convolutifs (CNN) permettent
d'automatiser une première identification à partir d'une photographie [1, 2].

### 1.2 Problématique
Comment concevoir un système fondé sur les CNN capable d'identifier la plante et la maladie à partir d'une image de feuille, avec une
précision suffisante pour constituer un **outil expérimental d'aide à l'identification** ?

### 1.3 Question scientifique
Parmi un CNN construit from scratch et trois architectures pré-entraînées (VGG16 [3], ResNet50 [4], MobileNetV2 [5]), laquelle offre le
meilleur compromis performance / taille / temps d'inférence, et dans quelle mesure se généralise-t-elle hors du jeu de données ?

### 1.4 Objectifs
Construire et comparer les modèles sur des métriques réelles ; ajouter l'explicabilité (Grad-CAM [6]) ; intégrer le modèle retenu dans une
application Web complète (API, interface, base de données) ; tester, sécuriser, conteneuriser et documenter l'ensemble.

## 2. Données

### 2.1 Jeu de données
PlantVillage [1, 2], variante couleur non augmentée : **54 305** images valides (aucune corrompue), 256×256 RGB, **38 classes** « Plante___Maladie »
sur 14 plantes (analyse : `docs/03-dataset.md`, produite par `ml.preprocessing.analyze_dataset`).

### 2.2 Déséquilibre et doublons
La plus petite classe compte 152 images (`Potato___healthy`), la plus grande 5 507 (`Orange___Haunglongbing_(Citrus_greening)`) : ratio **36,2**.
21 doublons exacts ont été détectés. Conséquence méthodologique : poids de classes et évaluation par **F1 macro** plutôt que par la seule accuracy.

### 2.3 Découpage sans fuite
Découpage 70/15/15 stratifié par classe, graine 42 ; les images de contenu identique (même hachage) sont regroupées dans le **même** sous-ensemble.

| Sous-ensemble | Images |
|---|---:|
| Entraînement | 38 014 |
| Validation | 8 147 |
| Test | 8 144 |

Vérification : 0 contenu présent dans plus d'un sous-ensemble. **Limite** : le hachage ne détecte que les doublons exacts ; des photos d'une même feuille
prises sous des angles différents peuvent se trouver dans plusieurs sous-ensembles et gonfler la performance.

## 3. Méthodologie

### 3.1 Prétraitement et augmentation
Images 224×224, valeurs conservées en [0, 255] ; la normalisation propre à chaque architecture est intégrée au modèle, ce qui garantit que l'entraînement et
l'inférence appliquent la même transformation. Augmentation (entraînement seul) : flips horizontal et vertical, rotation ±36°, zoom ±15 %, translation
10 %, luminosité et contraste ±15 %. Aucun changement de teinte : la couleur (jaunissement, taches) est le signal qui distingue les maladies.

### 3.2 Modèles
| Modèle | Paramètres totaux (mesurés, 38 classes) |
|---|---:|
| CNN baseline (4 blocs Conv-BN-ReLU-MaxPool, pooling global, Dropout 0,4) | 399 622 |
| MobileNetV2 (ImageNet) | 2 306 674 |
| ResNet50 (ImageNet) | 23 665 586 |
| VGG16 (ImageNet) | 14 734 194 |

Transfert : base pré-entraînée sans sa tête, puis pooling global, Dropout 0,3, couche Dense softmax.

### 3.3 Entraînement
Baseline : Adam 1e-3, poids de classes, arrêt anticipé sur la perte de validation, 10 époques maximum. Transfert : base gelée, tête seule. Fine-tuning :
dégel des 30 dernières couches de la base (BatchNorm figée), Adam 1e-5. Graine 42. La validation sert à choisir ; **le test n'est utilisé qu'une fois**, à la comparaison finale.

### 3.4 Évaluation
Accuracy, précision, rappel et **F1 macro** (critère principal), F1 pondéré, matrice de confusion normalisée, latence par image, taille du modèle, nombre de paramètres.

## 4. Conception et réalisation

### 4.1 Architecture logicielle
Frontend HTML/CSS/JS sans framework → nginx → API FastAPI → service d'inférence (modèle chargé une fois) → SQLite et stockage d'images. La logique métier est
dans des services, jamais dans les routes (voir `docs/ARCHITECTURE.md` et les diagrammes UML de `docs/02-uml.md`).
Le projet comprend environ 2 400 lignes de Python (backend, ML, scripts) et 360 lignes de JavaScript.

### 4.2 Sécurité
Validation d'upload en cinq temps (taille, extension, type MIME, contenu réel, dimensions) ; noms de fichiers générés (UUID) ; requêtes ORM paramétrées ;
aucun `innerHTML` côté client ; politique CSP stricte ; limitation de débit ; en-têtes de sécurité ; chargement du modèle en mode sûr avec empreinte SHA-256 ;
conteneurs sans privilèges et en lecture seule. Audit des dépendances (`pip-audit`) : aucune vulnérabilité connue après mise à jour de `pip` et `setuptools`.
Un test de sécurité a révélé et fait corriger une faille réelle : un petit PNG pouvait déclarer des dimensions énormes (« bombe de décompression ») et coûter
environ 240 Mo de mémoire au décodage. **Limite majeure : aucune authentification** (conforme au cahier des charges) : tout utilisateur peut lire et supprimer l'historique.

### 4.3 Tests
133 fonctions de test (plus d'exécutions grâce à la paramétrisation), couverture mesurée de **89 %** en Phase 16 : unitaires, API, intégration sur un vrai serveur HTTP,
sécurité, scripts d'entraînement de bout en bout, fichiers Docker, hygiène du dépôt et cohérence de la documentation. Limite : les tests ML utilisent des images
synthétiques et des modèles non entraînés ; ils vérifient le fonctionnement du code, **pas** la qualité des prédictions.

### 4.4 Déploiement
Deux images Docker (backend Python 3.11 de 3,1 Go dont 1,9 Go pour TensorFlow ; frontend nginx sans privilèges), `docker-compose.yml` avec systèmes de fichiers en lecture seule
et capacités retirées. Vérifié : construction, démarrage, validation d'upload, limitation de débit, persistance après redémarrage, interface sans violation de CSP.

## 5. Résultats

### 5.1 Performance d'inférence (mesurée, architecture à poids aléatoires)
Médiane de `predict()` pour une photo de 3000×2250 px (CPU 20 threads, machine non dédiée, charge inconnue) :

| Modèle | Médiane | p95 |
|---|---:|---:|
| CNN baseline | 105 ms | 142 ms |
| MobileNetV2 | 250 ms | 314 ms |
| ResNet50 | 365 ms | 462 ms |
| VGG16 | 212 ms | 241 ms |

Les quatre architectures respectent la cible de 2 s avec une marge d'un facteur 5 à 20 (`docs/18-performance.md`). Avec 8 clients simultanés, la latence p95 dépasse 2 s
(2,4 s) à cause de la sérialisation de l'inférence.

### 5.2 CNN baseline (validation uniquement, entraînement en cours)
Meilleure précision de **validation** relevée : 75,4 % à l'époque 5 (`ml/artifacts/baseline/history.csv`). Cette valeur n'est **pas** une performance de test et la
courbe de validation est irrégulière aux époques 3 et 4. *[À COMPLÉTER — Phase 8 : valeur finale et courbes d'apprentissage.]*

### 5.3 Comparaison des modèles sur le jeu de test
**[À COMPLÉTER — Phase 8]** Tableau accuracy / précision / rappel / F1 macro / latence / taille, issu de `docs/results/comparison.md`, et matrices de confusion
(`docs/results/confusion_*.png`). Les classes les plus confondues seront discutées d'après les matrices, pas avant.

### 5.4 Explicabilité
**[À COMPLÉTER]** Cartes Grad-CAM sur un modèle entraîné, à inspecter visuellement : le modèle s'appuie-t-il sur la tache de maladie ou sur le fond ?

### 5.5 Écart entre entraînement et inférence
Le redimensionnement diffère (TensorFlow à l'entraînement, Pillow à l'inférence) : écart moyen de 1,0/255 par pixel sur 20 images réelles. Son effet sur les prédictions
d'un modèle entraîné reste **à mesurer** *[À COMPLÉTER]*.

## 6. Discussion et limites

- **Généralisation non mesurée.** PlantVillage présente des feuilles isolées sur fond uniforme. Une bonne performance sur le test n'implique pas une bonne performance
  sur des photographies prises au champ ; aucun jeu de test indépendant n'existe dans ce projet. C'est la limite scientifique principale.
- **Confiance non calibrée.** Le seuil de confiance faible (0,6) est arbitraire ; une probabilité softmax élevée ne garantit pas l'exactitude ; le modèle n'a pas de classe
  « autre » et prédit donc une maladie même pour une image qui n'est pas une feuille.
- **Fuites résiduelles possibles** entre sous-ensembles (même feuille, angles différents).
- **Une seule graine par modèle** : pas d'intervalle de confiance sur les métriques.
- **Base de connaissances partielle** : fiches rédigées pour 6 classes sur 38 ; à faire valider par une source agronomique avant tout usage.
- **Contraintes de calcul.** Entraînement sur CPU, sans GPU. Le processus d'entraînement a été interrompu à plusieurs reprises par un épuisement de mémoire du poste
  (machine virtuelle WSL de Docker Desktop, base de données, entraînement), jusqu'à un redémarrage de Windows ; l'entraînement a été rendu reprenable (point de reprise à chaque
  époque) et les étapes sont exécutées une à la fois. Cette difficulté est documentée plutôt que masquée car elle a limité le nombre d'expériences réalisées.
- **Outil expérimental** : à ne pas utiliser pour une décision agronomique sans avis d'un professionnel.

## 7. Conclusion et perspectives
*[À COMPLÉTER — Phase 8 : réponse à la question scientifique d'après la comparaison, et conditions dans lesquelles le résultat peut être cru.]*

Perspectives (non réalisées) : jeu de test hors dataset (photos au champ), calibration de la confiance et classe « autre », authentification et quotas, plusieurs graines
avec intervalles de confiance, quantification (TFLite) pour un déploiement léger.

## Références (à vérifier avant remise)
1. D. P. Hughes, M. Salathé, *An open access repository of images on plant health to enable the development of mobile disease diagnostics*, arXiv:1511.08060, 2015.
2. S. P. Mohanty, D. P. Hughes, M. Salathé, *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science, 7:1419, 2016.
3. K. Simonyan, A. Zisserman, *Very deep convolutional networks for large-scale image recognition*, ICLR, 2015.
4. K. He, X. Zhang, S. Ren, J. Sun, *Deep residual learning for image recognition*, CVPR, 2016.
5. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, *MobileNetV2: inverted residuals and linear bottlenecks*, CVPR, 2018.
6. R. R. Selvaraju et al., *Grad-CAM: visual explanations from deep networks via gradient-based localization*, ICCV, 2017.

## Annexes
A. Documentation du dépôt : `docs/` (une page par phase, plus `ARCHITECTURE.md`, `API.md`, `MODEL.md`, `INSTALLATION.md`).
B. Reproduction : `docs/INSTALLATION.md` et `python -m scripts.run_pipeline`.
