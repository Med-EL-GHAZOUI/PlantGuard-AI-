# Phase 19 — Docker

## Lancer
```bash
python -m ml.export.export_model --model <best.keras> --name <nom>   # produit ml/artifacts/final/
docker compose up --build                                            # http://localhost:8080
```
Variables facultatives : `MODEL_DIR` (défaut `./ml/artifacts/final`), `FRONTEND_PORT` (défaut `8080`),
`ENABLE_DOCS` (défaut `0`). Arrêt : `docker compose down` (ajouter `-v` pour **supprimer** l'historique et les images stockées).
Le modèle n'est **pas** dans l'image (lourd, non versionné) : il est monté en lecture seule sur `/models`.
Sans modèle, l'API démarre en mode dégradé (bandeau dans l'interface, `/api/predict` répond 503).

## Architecture
```text
navigateur ──► nginx :8080 ──► backend :8000 (réseau interne, aucun port publié sur l'hôte)
              (frontend statique,        │
               CSP, limite de débit)     ├─ /models  (modèle, lecture seule)
                                         └─ /data    (volume : SQLite + images)
```
Frontend et API partagent **la même origine** (nginx fait proxy de `/api/`) : plus de CORS à ouvrir.

## Fichiers
| Fichier | Rôle |
|---|---|
| `docker/backend.Dockerfile` | Python 3.11-slim, dépendances figées, seul le code d'inférence est copié, utilisateur `app` (uid 10001), `HEALTHCHECK` |
| `docker/frontend.Dockerfile`, `nginx.conf`, `proxy.conf`, `config.js` | nginx **sans privilèges**, CSP stricte, `server_tokens off`, 6 Mo max, limite de débit par client |
| `docker-compose.yml` | systèmes de fichiers en lecture seule, `cap_drop: ALL`, `no-new-privileges`, volume de données, healthcheck |
| `backend/requirements.txt` | dépendances d'exécution **figées** (dont TensorFlow 2.21.0) ; `requirements-dev.txt` ajoute entraînement et tests |
| `.dockerignore` | exclut dataset, artefacts, venv, `.git`, tests |

## Vérifié pour de vrai (images construites, pile lancée, modèle non entraîné)
| Contrôle | Résultat |
|---|---|
| `docker compose build` / `up` | 2 images construites ; backend `healthy` ; frontend démarre après lui |
| Prédiction via nginx | 200 ; image `.txt` → 415 ; fichier de 7 Mo → 413 |
| En-têtes | CSP, `nosniff`, `X-Frame-Options`, `Referrer-Policy` présents **une seule fois** ; `Server: nginx` sans version |
| Limitation de débit (nginx) | 16 envois rapides : 200 pendant la rafale, puis 429 ; les lectures restent libres |
| Backend non joignable depuis l'hôte | seul `8000/tcp` interne, aucun port publié |
| Processus | backend uid 10001 ; écriture refusée dans `/app` et `/models`, autorisée dans `/data` uniquement |
| Persistance | 14 analyses avant et après `restart` ; image toujours servie |
| Interface dans le navigateur, via nginx | analyse, historique (images chargées), dashboard, 38 maladies, **0 violation de CSP** |
| Tests automatiques | 8 tests statiques sur les fichiers Docker ; suite complète : **118 tests passent** |

## Problèmes rencontrés et corrigés
1. `.dockerignore` excluait `frontend/` alors que l'image nginx en a besoin (contexte de build partagé).
2. `RUN printf > config.js` échouait : l'image nginx « unprivileged » ne peut pas écrire dans son dossier web → fichier fourni par `COPY`.
3. La limitation de débit applicative (par adresse) verrait toutes les requêtes venir de nginx : elle est désactivée dans compose (`=0`) et faite par nginx, par client réel.
4. `api.js` traitait l'adresse vide `""` (même origine) comme « absente » à cause de `||` : remplacé par `??` (testé).

## Limites
- **Image backend : 3,1 Go** (TensorFlow seul : 1,9 Go). Acceptable pour un projet académique ; réduire demanderait un autre moteur d'inférence (TFLite/ONNX).
- Testé avec un modèle **non entraîné** : la chaîne Docker est validée, pas la qualité des prédictions.
- Un seul worker et un seul conteneur backend : limites de débit décrites en Phase 18.
- Pas de HTTPS : à ajouter devant nginx (reverse proxy / certificat) pour un déploiement public.
- Testé sous Windows (Docker Desktop) uniquement ; build non testé sur ARM.
- Les images de développement sont conservées localement (`applearning-backend`, `applearning-frontend`) : `docker rmi applearning-backend applearning-frontend` pour les supprimer.
