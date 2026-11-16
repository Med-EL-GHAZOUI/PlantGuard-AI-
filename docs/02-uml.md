# Phase 2 — Diagrammes UML

Les sources PlantUML sont dans `docs/uml/`. Rendu : extension VS Code « PlantUML » ou https://www.plantuml.com/plantuml.

| Diagramme | Fichier | Objectif | Acteurs / éléments | Relations clés |
|---|---|---|---|---|
| Use Case | `use_case.puml` | Fonctions offertes à chaque acteur | Utilisateur, Développeur | include (upload→validation, preview), extend (résultat→infos maladie) |
| Classes | `class.puml` | Structure des données et services | Analysis, Disease, services, repositories | Analysis → Disease (N:1), PredictionService → ModelLoader |
| Séquence | `sequence.puml` | Déroulé de `POST /api/predict` | Utilisateur, Frontend, API, Validator, Service, Modèle, DB | Branche alt : fichier invalide / valide |
| Activité | `activity.puml` | Flux logique d'une analyse | Utilisateur, système | Décisions : validation fichier, lecture image |
| Composants | `component.puml` | Modules et dépendances | Frontend, Backend, ML, DB | Frontend → Routers → Services → ML / DB |
| Déploiement | `deployment.puml` | Exécution Docker | Navigateur, conteneurs, volume | Frontend → Backend → Volume |
