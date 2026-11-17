# 🌿 PlantGuard AI

Application Web qui analyse la photo d'une feuille et prédit la **plante** et la **maladie** éventuelle grâce à
un réseau de neurones convolutif (CNN / Transfer Learning).

> ⚠️ **Outil expérimental d'aide à l'identification, pas un diagnostic professionnel.**
> Le dataset (PlantVillage) est photographié sur fond uniforme : la performance en conditions réelles
> sera probablement inférieure à celle mesurée sur le jeu de test.

## État du projet (honnête)

| Élément | État |
|---|---|
| Pipeline de données (analyse, split sans fuite, prétraitement, augmentation) | ✅ fait, mesuré sur 54 305 images réelles |
| Code des modèles (CNN baseline, VGG16, ResNet50, MobileNetV2), fine-tuning, comparaison, Grad-CAM | ✅ écrit et testé sur données synthétiques |
| **Entraînement des modèles** | ⏳ **en cours, interrompu à plusieurs reprises par un manque de mémoire du poste** : baseline à 5/10 époques (75,4 % de précision en **validation**, relevé du 17/11 à 02:53) ; les 3 modèles de transfert ne sont **pas encore entraînés** |
| **Comparaison sur le jeu de test, choix du modèle final** | ⏳ **pas encore faite** : aucune performance finale n'est annoncée ici tant qu'elle n'est pas mesurée |
| Backend FastAPI, base SQLite, frontend, Docker, sécurité | ✅ fait et vérifié (avec un modèle **non entraîné**) |
| Rapport académique, présentation | ⏳ à faire |

## Fonctionnalités
Upload d'image (aperçu, glisser-déposer) · analyse et résultat (plante, maladie, confiance, top-3, alerte de
confiance faible) · fiches maladies · historique (détail, suppression) · dashboard de statistiques · API REST.

## Architecture
```text
navigateur ──► frontend (HTML/CSS/JS) ──► FastAPI ──► PredictionService ──► modèle Keras
                                              └──► SQLite (analyses) + images (noms UUID)
```
La logique métier est dans `backend/app/services/`, jamais dans `main.py`.

## Démarrage rapide

### Option A — Docker (recommandé)
```bash
python -m ml.export.export_model --model <chemin/best.keras> --name <nom>   # produit ml/artifacts/final/
docker compose up --build                                                    # http://localhost:8080
```
Sans modèle exporté l'application démarre en mode dégradé (l'analyse répond 503).

### Option B — local
```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m scripts.run_dev                                 # API :8000 + frontend :5500
python -m scripts.run_dev --api-port 8100 --web-port 5600 # si un port est occupé
```

## Pipeline ML
```bash
python -m ml.dataset.download_dataset                                               # PlantVillage (Kaggle, public)
python -m ml.preprocessing.analyze_dataset --data-dir ml/dataset/raw --output-dir docs/figures
python -m ml.preprocessing.split_dataset --data-dir ml/dataset/raw --output-dir ml/dataset/split
python -m ml.preprocessing.class_info --train-dir ml/dataset/split/train --output-dir ml/artifacts
python -m ml.training.train_baseline --epochs 10                 # --resume pour reprendre après interruption
python -m ml.training.train_transfer --model mobilenetv2 --epochs 5
python -m ml.training.train_finetune --model mobilenetv2 --unfreeze 30 --epochs 5
python -m ml.evaluation.compare_models --model nom=chemin.keras ...                 # jeu de TEST
python -m ml.explainability.gradcam --model chemin.keras --image feuille.jpg
```
Le jeu de test n'est utilisé que par `compare_models` (mesure honnête). Les poids ImageNet sont téléchargés par Keras au premier lancement.

## API
| Route | Rôle |
|---|---|
| `GET /api/health` | état de l'API et du modèle |
| `POST /api/predict` | image (`multipart`, champ `file`) → plante, maladie, confiance, top-3 |
| `GET /api/analyses` · `GET /api/analyses/{id}` · `…/image` | historique paginé, détail, image |
| `DELETE /api/analyses/{id}` | supprime l'analyse et son image |
| `GET /api/stats` · `GET /api/diseases` | statistiques ; classes supportées et fiches |

Documentation interactive sur `/docs` (désactivable : `PLANTGUARD_ENABLE_DOCS=0`).
Variables d'environnement : voir [.env.example](.env.example).

## Tests
```bash
python -m pytest                                  # 128 tests (~3 min, surtout TensorFlow)
python -m pytest --cov --cov-report=term-missing  # couverture mesurée : 89 % (Phase 16)
```
Les tests ML utilisent des images synthétiques et des modèles non entraînés : ils vérifient que le code
fonctionne, **pas** que les modèles prédisent bien. La CI GitHub Actions (`.github/workflows/ci.yml`) lance
les tests, l'audit des dépendances et la construction Docker.

## Structure du dépôt
```text
backend/app/   api · services · schemas · models · database · utils · data (fiches maladies)
frontend/      index.html · css · js (modules, sans framework)
ml/            dataset · preprocessing · models · training · evaluation · explainability · export · artifacts
scripts/       run_dev.py (lance API + frontend)
docker/        Dockerfiles, nginx
docs/          une page par phase (voir ci-dessous) · figures · results · uml
tests/         pytest
```

## Sécurité (résumé)
Validation d'upload en 5 temps (taille, extension, MIME, contenu réel, dimensions) · noms de fichiers UUID ·
requêtes ORM paramétrées · aucun `innerHTML` · CSP stricte · limitation de débit · en-têtes de sécurité ·
chargement du modèle en mode sûr + empreinte SHA-256 · conteneurs non-root en lecture seule · 0 vulnérabilité
connue dans les dépendances (`pip-audit`). **Limite majeure : pas d'authentification** (conforme au cahier
des charges) : toute personne joignant l'API peut lire et supprimer l'historique. Détails : [docs/17-securite.md](docs/17-securite.md).

## Documentation
| Phase | Document |
|---|---|
| 0 Analyse | [00-analyse-projet.md](docs/00-analyse-projet.md) |
| 1 Cahier des charges | [01-cahier-des-charges.md](docs/01-cahier-des-charges.md) |
| 2 UML | [02-uml.md](docs/02-uml.md) |
| 3 Dataset | [03-dataset.md](docs/03-dataset.md) |
| 4 Prétraitement | [04-pretraitement.md](docs/04-pretraitement.md) |
| 5 CNN baseline | [05-cnn-baseline.md](docs/05-cnn-baseline.md) |
| 6 Transfer Learning | [06-transfer-learning.md](docs/06-transfer-learning.md) |
| 7 Fine-tuning | [07-fine-tuning.md](docs/07-fine-tuning.md) |
| 8 Comparaison | [08-comparaison.md](docs/08-comparaison.md) |
| 9 Grad-CAM | [09-gradcam.md](docs/09-gradcam.md) |
| 10-11 Export et inférence | [10-11-export-inference.md](docs/10-11-export-inference.md) |
| 12 Backend | [12-backend.md](docs/12-backend.md) |
| 13 Base de données | [13-base-de-donnees.md](docs/13-base-de-donnees.md) |
| 14 Frontend | [14-frontend.md](docs/14-frontend.md) |
| 15 Intégration | [15-integration.md](docs/15-integration.md) |
| 16 Tests | [16-tests.md](docs/16-tests.md) |
| 17 Sécurité | [17-securite.md](docs/17-securite.md) |
| 18 Performance | [18-performance.md](docs/18-performance.md) |
| 19 Docker | [19-docker.md](docs/19-docker.md) |
| 20 Git et dépôt | [20-git.md](docs/20-git.md) |
| 21 Documentation finale | [ARCHITECTURE.md](docs/ARCHITECTURE.md) · [API.md](docs/API.md) · [MODEL.md](docs/MODEL.md) · [INSTALLATION.md](docs/INSTALLATION.md) |

## Limites connues
Fond uniforme du dataset (généralisation non mesurée hors dataset) · base de connaissances limitée à 6 classes sur 38 ·
image Docker de 3,1 Go · débit limité à une prédiction à la fois par processus · pas de HTTPS ni d'authentification.

## Licence
À définir par l'auteur : aucune licence n'est encore choisie, donc **tous droits réservés** par défaut.
