# Phase 1 — Cahier des charges : PlantGuard AI

## 1. Présentation du projet
PlantGuard AI est une application Web qui analyse la photo d'une feuille et prédit la plante et la
maladie éventuelle grâce à un modèle de Deep Learning (CNN / Transfer Learning). Outil
**expérimental d'aide à l'identification**, pas un diagnostic professionnel.

## 2. Contexte
Les maladies foliaires causent des pertes agricoles ; l'expertise humaine est coûteuse et peu
disponible. La vision par ordinateur offre une première identification rapide et accessible.

## 3. Problématique
Comment concevoir un système basé sur les CNN capable d'identifier automatiquement les maladies des
plantes à partir d'images de feuilles avec une précision suffisante pour constituer un outil
expérimental d'aide à l'identification ?

## 4. Objectifs
- Modèle de classification plante + maladie, choisi par comparaison expérimentale réelle.
- Application Web complète : API, interface, base de données, dashboard.
- Projet testé, documenté, conteneurisé et versionné sur GitHub.

## 5. Fonctionnalités
| Code | Fonctionnalité | Priorité |
|---|---|---|
| F1 | Upload d'image (sélection, prévisualisation, drag & drop) | Haute |
| F2 | Analyse Deep Learning (prétraitement + prédiction) | Haute |
| F3 | Affichage du résultat (plante, maladie, confiance, statut) | Haute |
| F4 | Base de connaissances sur les maladies | Moyenne |
| F5 | Historique des analyses | Moyenne |
| F6 | Dashboard de statistiques | Moyenne |
| F7 | API REST | Haute |
| F8 | Explicabilité Grad-CAM (si possible) | Basse |

## 6. Utilisateurs
Utilisateur final (photo → résultat) · Évaluateur/enseignant (API, dashboard, résultats) ·
Développeur (entraînement, maintenance). Pas d'authentification dans la version initiale.

## 7. Besoins fonctionnels
- BF1 : envoyer une image JPG/JPEG/PNG et la prévisualiser avant envoi.
- BF2 : valider type MIME, extension, taille et validité de l'image.
- BF3 : retourner `{plant, disease, confidence}` en JSON.
- BF4 : afficher image, plante, maladie, confiance (%) et informations associées.
- BF5 : enregistrer chaque analyse (date, image, plante, maladie, confiance).
- BF6 : consulter, détailler et supprimer les analyses passées.
- BF7 : afficher total d'analyses, maladies détectées, plantes les plus analysées,
  statistiques par maladie, évolution dans le temps.
- BF8 : exposer `POST /api/predict`, `GET /api/analyses`, `GET /api/analyses/{id}`,
  `DELETE /api/analyses/{id}`, `GET /api/diseases`, `GET /api/stats`, `GET /api/health`.
- BF9 : avertissement visible « outil expérimental, pas un diagnostic professionnel ».

## 8. Besoins non fonctionnels
- **Performance :** inférence < 2 s par image sur CPU (cible à mesurer, non garantie).
- **Fiabilité :** erreurs gérées avec codes HTTP et messages clairs.
- **Maintenabilité :** code modulaire (routers / services / database), noms cohérents.
- **Portabilité :** exécution locale et via Docker.
- **Utilisabilité :** interface responsive (mobile et desktop).
- **Reproductibilité :** versions figées dans `requirements.txt`, graines aléatoires fixées.

## 9. Contraintes techniques
Frontend HTML5/CSS3/JS vanilla · Backend Python, FastAPI, Uvicorn, Pydantic · ML TensorFlow/Keras,
OpenCV, Pillow · Données NumPy/Pandas/Matplotlib/scikit-learn · SQLite (migration PostgreSQL
possible) · ressources de calcul limitées · dataset de type PlantVillage.

## 10. Contraintes de sécurité
- Validation extension + MIME + signature réelle de l'image (ouverture avec Pillow).
- Taille maximale d'upload (ex. 5 Mo) et noms de fichiers générés (UUID), jamais ceux de l'utilisateur.
- Aucun secret dans le code : variables d'environnement (`.env` ignoré par Git).
- Frontend et backend séparés ; CORS restreint.
- Gestion d'erreurs sans fuite d'informations internes.

## 11. Architecture
```text
Frontend (HTML/JS) → Fetch API → FastAPI → Prediction Service → Modèle TensorFlow
                                       ↘ Database (SQLite)
```
Arborescence : `frontend/`, `backend/app/{api,models,schemas,services,database,utils}`, `ml/`, `tests/`,
`docs/`. Règle : la logique métier est dans `services/`, jamais dans `main.py`.

## 12. Technologies
Voir §9. Modèles étudiés : CNN personnalisé, VGG16, ResNet50, MobileNetV2.

## 13. Planning (indicatif)
| Étape | Contenu |
|---|---|
| 1 | Phases 0–2 : analyse, cahier des charges, UML |
| 2 | Phases 3–4 : dataset, prétraitement |
| 3 | Phases 5–8 : CNN, Transfer Learning, fine-tuning, comparaison |
| 4 | Phases 9–11 : Grad-CAM, sauvegarde, inférence |
| 5 | Phases 12–15 : backend, base de données, frontend, intégration |
| 6 | Phases 16–19 : tests, sécurité, performance, Docker |
| 7 | Phases 20–23 : Git, documentation, rapport, présentation |

## 14. Livrables
Code source (GitHub) · modèle entraîné + fichier de configuration · application Web · tests ·
documentation (README, ARCHITECTURE, API, MODEL, INSTALLATION) · diagrammes UML ·
rapport académique · présentation.

## 15. Critères de réussite
- Pipeline complet fonctionnel : upload → prédiction → enregistrement → affichage.
- Comparaison de ≥ 4 modèles avec métriques réelles sur un jeu de test jamais vu à l'entraînement.
- Aucune fuite de données entre train / validation / test.
- Tests automatisés passent ; `docker compose up --build` démarre l'application.
- Documentation permettant à un tiers d'installer et lancer le projet.
- Le seuil de performance cible sera fixé après analyse du dataset (Phase 3), pas avant.
