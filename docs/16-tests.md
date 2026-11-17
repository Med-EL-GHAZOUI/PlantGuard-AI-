# Phase 16 — Tests

## Exécution
```bash
python -m pytest                                 # tous les tests (~3 min, surtout TensorFlow)
python -m pytest --cov --cov-report=term-missing # avec couverture
python -m pytest tests/test_api.py tests/test_security.py   # partie rapide, sans TensorFlow
```

## Bilan mesuré : 92 tests, couverture 89 %
(`ml/dataset/download_dataset.py` exclu : il appelle Kaggle et le réseau.)

| Fichier | Couvre |
|---|---|
| `test_dataset_scripts.py`, `test_preprocessing.py` | analyse du dataset, split sans fuite, prétraitement, poids de classes, augmentation |
| `test_baseline.py`, `test_transfer.py`, `test_finetune.py` | architectures, gel des bases, équivalence du prétraitement avec Keras, dégel partiel |
| `test_training_scripts.py` | **les 3 scripts d'entraînement de bout en bout** (1 époque, mini-dataset synthétique) |
| `test_compare.py`, `test_gradcam.py`, `test_export_and_service.py` | métriques, Grad-CAM, export, service d'inférence |
| `test_api.py`, `test_analyses_api.py` | routes, validation d'upload, historique, statistiques |
| `test_security.py` | bombe de décompression, identifiants non entiers, injection SQL dans un nom de fichier, mauvaises méthodes HTTP, configuration |
| `test_ml_robustness.py` | détecteur de fuite testé sur une vraie fuite, reproductibilité du split, classes minuscules, fichiers corrompus |
| `test_integration.py` | vrai serveur HTTP, requêtes concurrentes, CORS |

## Ce que cette phase a trouvé
1. **Faille corrigée — bombe de décompression** : un petit PNG pouvait déclarer jusqu'à ~89 millions de pixels sans que Pillow ne réagisse, et le décodage aurait coûté ~240 Mo par requête. `validate_upload` lit maintenant les dimensions dans l'en-tête et refuse au-delà de 25 millions de pixels (413). Le test échoue sans la correction (vérifié).
2. **Scripts jamais exécutés** : `train_transfer` et `train_finetune` n'avaient jamais tourné ; ils fonctionnent (le fine-tuning augmente bien le nombre de paramètres entraînables).

## Limites (honnêtes)
- Tous les tests ML utilisent des images aléatoires et des modèles non entraînés : ils prouvent que le code s'exécute et que les contrats sont respectés, **pas** que les modèles prédisent bien. Seule la Phase 8 sur le vrai jeu de test le mesure.
- Pas de tests automatisés du JavaScript du frontend (vérification manuelle en Phase 14).
- Non couvert (89 %) : les fonctions `main()` en ligne de commande et `run_dev.main`.
- Le test de concurrence utilise 6 threads ; il ne remplace pas un test de charge.
