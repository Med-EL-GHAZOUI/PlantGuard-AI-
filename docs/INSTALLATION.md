# Installation — PlantGuard AI

## Prérequis
| Besoin | Détail |
|---|---|
| Python | **3.11** (version testée ; d'autres versions non vérifiées) |
| Docker | Docker Engine + Compose v2 (testé avec Docker 29.6 sous Windows) : seulement pour l'option Docker |
| Espace disque | l'image Docker backend fait **3,1 Go** (TensorFlow seul : 1,9 Go) ; prévoir en plus le dataset et les modèles |
| Calcul | CPU suffisant pour l'inférence (< 0,5 s par image, mesuré) ; l'**entraînement sur CPU est long** (~16 min par époque du baseline sur un CPU de 20 threads). TensorFlow ≥ 2.11 n'utilise pas le GPU sous Windows natif |
| Réseau | téléchargement du dataset (Kaggle, public) et des poids ImageNet (Keras) au premier entraînement de transfert |

## 1. Récupérer le projet
```bash
git clone https://github.com/Med-EL-GHAZOUI/PlantGuard-AI-.git
cd PlantGuard-AI-
```

## 2. Environnement Python
```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt  # exécution + entraînement + tests (versions figées)
```
Pour l'exécution seule : `pip install -r backend/requirements.txt`.

## 3. Obtenir un modèle
Le dépôt **ne contient aucun modèle entraîné** (fichiers `.keras` ignorés par Git). Il faut en produire un :
```bash
python -m ml.dataset.download_dataset                                                     # PlantVillage dans ml/dataset/raw
python -m ml.preprocessing.split_dataset --data-dir ml/dataset/raw --output-dir ml/dataset/split
python -m ml.preprocessing.class_info --train-dir ml/dataset/split/train --output-dir ml/artifacts
python -m ml.training.train_baseline --epochs 10                       # ou train_transfer / train_finetune
python -m ml.export.export_model --model ml/artifacts/baseline/best.keras --name baseline
```
Pour enchaîner **une étape à la fois** (baseline → MobileNetV2 → fine-tuning → comparaison sur le test), reprenable et sans calcul en parallèle : `python -m scripts.run_pipeline` ; sa progression est dans `ml/artifacts/pipeline_status.json`. Il s'arrête avant l'export : le choix du modèle final reste humain.

`export_model` écrit `ml/artifacts/final/` (modèle, `class_names.json`, `model_config.json`). Pour reprendre un entraînement
interrompu du baseline : `python -m ml.training.train_baseline --resume --epochs 10`.
Pour comparer les modèles sur le test : voir [MODEL.md](MODEL.md).

## 4a. Lancer avec Docker (recommandé)
```bash
docker compose up --build            # http://localhost:8080
```
Variables facultatives (shell ou fichier `.env`) : `MODEL_DIR` (défaut `./ml/artifacts/final`), `FRONTEND_PORT` (défaut `8080`),
`ENABLE_DOCS` (défaut `0`). Arrêt : `docker compose down` ; ajouter `-v` **supprime l'historique et les images stockées**.

## 4b. Lancer sans Docker
```bash
python -m scripts.run_dev            # interface : http://127.0.0.1:5500   API : http://127.0.0.1:8000/docs
```
Ports occupés : `python -m scripts.run_dev --api-port 8100 --web-port 5600`. Autre dossier de modèle : `--model-dir chemin`.

## 5. Vérifier l'installation
```bash
curl http://localhost:8080/api/health      # Docker ; sans Docker : http://127.0.0.1:8000/api/health
python -m pytest                           # suite complète (~3 min)
```
Réponse attendue de `/api/health` : `{"status":"ok","model_loaded":true,...}`.

## Configuration
Toutes les variables sont listées dans [.env.example](../.env.example) (`PLANTGUARD_MODEL_DIR`, `PLANTGUARD_DATABASE_URL`,
`PLANTGUARD_UPLOAD_DIR`, `PLANTGUARD_MAX_UPLOAD_MB`, `PLANTGUARD_RATE_LIMIT_PER_MINUTE`, `PLANTGUARD_ENABLE_DOCS`, `PLANTGUARD_CORS_ORIGINS`).

## Dépannage
| Symptôme | Cause et remède |
|---|---|
| `Le port 8000 (API) est déjà utilisé` | un autre programme occupe le port : `--api-port` / `--web-port` (testé : le message apparaît bien) |
| Bandeau « le modèle n'est pas chargé » ; `/api/predict` répond 503 | aucun modèle exporté : étape 3 ; vérifier `PLANTGUARD_MODEL_DIR` / `MODEL_DIR` |
| `Fichiers manquants dans … model.keras, class_names.json, model_config.json` | export incomplet : relancer `ml.export.export_model` |
| `Empreinte du modèle différente de model_config.json` | `model.keras` modifié après l'export : le réexporter |
| `Chargement du modèle impossible ou refusé` | fichier corrompu, ou contenant du code arbitraire (couche `Lambda`), refusé volontairement |
| `Reprise impossible : aucun point de reprise` | pas de `best.keras` / `last.keras` ni d'historique : lancer sans `--resume` |
| L'entraînement s'arrête sans message, ou `AbortedError … could not create a memory object`, ou Windows redémarre | **mémoire insuffisante** : l'entraînement veut 4 à 6 Go ; sur un poste de 16 Go partagé avec Docker Desktop (sa VM WSL peut en prendre ~10), cela sature. Fermer Docker Desktop ou limiter sa VM (`%UserProfile%\.wslconfig`, `[wsl2] memory=4GB`, puis `wsl --shutdown`), ne pas lancer d'autre calcul TensorFlow en parallèle, puis `--resume` |
| Avertissement TensorFlow « GPU support is not available on native Windows » | normal : l'exécution se fait sur CPU |
| `docker compose build` très long la première fois | installation de TensorFlow (~2 Go) ; les couches suivantes sont en cache |
| L'interface charge mais n'atteint pas l'API (hors Docker) | l'origine du frontend doit figurer dans `PLANTGUARD_CORS_ORIGINS` (`run_dev` s'en charge) |
| 429 `Trop de requêtes` | limitation de débit (30/min) : attendre `Retry-After` secondes |

## Désinstallation
```bash
docker compose down -v                                  # conteneurs, réseau et volume de données
docker rmi applearning-backend applearning-frontend     # images (le préfixe dépend du nom du dossier du projet)
```
