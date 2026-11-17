# Phase 17 — Sécurité

## Audit réalisé
| Contrôle | Résultat |
|---|---|
| Secrets dans le dépôt et l'historique Git | Aucun ; `.env` ignoré ; aucune clé dans le code |
| Vulnérabilités des dépendances (`pip-audit`) | 21 alertes initiales, **toutes** sur `pip` et `setuptools` (outils de la venv, pas des dépendances d'exécution). Mis à jour → **0 vulnérabilité connue** |
| Validation d'upload | taille, extension, MIME, contenu réel, **dimensions** (bombe de décompression, Phase 16) |
| Injection SQL | requêtes ORM paramétrées ; identifiants typés `int` (422 sinon) ; testé avec un nom de fichier piégé |
| XSS | aucun `innerHTML` ; tout texte serveur via `textContent` (testé automatiquement) |
| Traversée de chemin | noms de fichiers générés (UUID) ; les images ne sont servies que via un identifiant en base |

## Durcissements ajoutés
| Mesure | Détail |
|---|---|
| En-têtes de sécurité (API) | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` |
| Limitation de débit | `POST /api/predict` et `DELETE /api/analyses/{id}` : 30 req/min/client par défaut (`PLANTGUARD_RATE_LIMIT_PER_MINUTE`, 0 = désactivé), réponse 429 + `Retry-After`, fenêtre glissante, thread-safe, mémoire bornée |
| CSP stricte (frontend) | `script-src 'self'; style-src 'self'; connect-src <API>` sans `unsafe-inline` ; `frame-ancestors 'none'`. Les 2 styles inline du frontend ont été remplacés par des classes CSS |
| Docs API masquables | `PLANTGUARD_ENABLE_DOCS=0` supprime `/docs` et `/openapi.json` en production |
| Chargement du modèle | `safe_mode=True` explicite : un `.keras` contenant une couche `Lambda` (code arbitraire) est refusé **même avec une empreinte SHA-256 valide** ; erreur convertie en `ModelLoadError` |

## Vérifié pour de vrai
- Dans le navigateur, avec la CSP active : parcours complet sans aucune violation ; et la CSP **bloque bien** un script inline, un style inline et une requête vers un hôte tiers (violations `script-src-elem`, `style-src-attr`, `connect-src` observées).
- Sans `safe_mode`, le fichier Lambda de test se chargerait : le test est donc utile.
- 105 tests passent.

## Limites connues (à assumer dans le rapport)
- **Pas d'authentification** (conforme au cahier des charges) : toute personne pouvant joindre l'API peut lire et **supprimer** l'historique de tous. La limitation de débit ne remplace pas un contrôle d'accès.
- La limitation de débit est **en mémoire, par processus** : avec plusieurs workers ou derrière un proxy (toutes les requêtes semblent venir de la même adresse), elle doit être faite par le proxy.
- Le stockage des images n'a pas de quota : seul le débit limite la croissance du disque.
- La CSP du frontend est posée par le serveur de dev (`scripts/run_dev.py`) ; en production, le serveur web (nginx, Phase 19) devra envoyer les mêmes en-têtes.
- HTTPS n'est pas géré par l'application : à terminer au niveau du proxy.
- L'intégrité du modèle (SHA-256) protège contre une corruption, pas contre un attaquant pouvant écrire à la fois le modèle **et** sa configuration.
