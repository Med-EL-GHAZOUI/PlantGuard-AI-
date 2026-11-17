# Phase 14 — Frontend (HTML / CSS / JS vanilla)

## Lancement
```bash
# 1. Backend (port 8000)
uvicorn backend.app.main:create_app --factory --reload
# 2. Frontend (port 5500 : origine autorisée par défaut par le CORS du backend)
python -m http.server 5500 --directory frontend
```
Ouvrir `http://127.0.0.1:5500` (ou `localhost:5500`). Autre adresse d'API : modifier `frontend/config.js` ;
autre origine du frontend : variable `PLANTGUARD_CORS_ORIGINS` côté backend.

## Pages (une seule page, navigation par `#hash`)
| Vue | Contenu |
|---|---|
| Analyser | sélection ou glisser-déposer, prévisualisation, contrôle de format/taille côté client, résultat (plante, maladie, confiance, statut, top-3, fiche maladie, avertissement) |
| Historique | liste paginée, détails (dialogue), suppression avec confirmation |
| Dashboard | totaux, maladies détectées, plantes les plus analysées, courbe par jour (SVG sans bibliothèque) |
| Maladies | classes supportées avec recherche, fiche si elle existe |

## Choix de conception
- **Aucun `innerHTML`** : tout texte venant du serveur passe par `textContent` (`js/dom.js`) → pas de XSS par nom de maladie ou message d'erreur.
- Contrôle côté client = confort ; le **serveur reste l'autorité** (il revalide tout).
- Messages d'erreur clairs (serveur injoignable, modèle non chargé, format refusé, fichier trop gros).
- Avertissement « outil expérimental » visible en permanence ; message dédié quand la confiance est faible.
- Accessible : `aria-live` sur le résultat, zone de dépôt au clavier, `<dialog>` natif. Responsive (une colonne sous 700 px).

## Vérification réalisée
Testé dans le navigateur avec le vrai backend et un modèle **non entraîné** (résultats sans signification) :
upload → prédiction → historique → détails → suppression → dashboard → liste des 38 maladies ; refus d'un
fichier `.txt` ; pas de défilement horizontal à 375 px.

## Limites
- Pas de tests automatisés du JavaScript (vérification manuelle uniquement).
- Les images de l'historique sont servies par l'API : sans authentification, toute personne ayant l'URL les voit.
- Les noms de maladies restent en anglais (ceux du dataset).
