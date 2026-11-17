# Phase 18 — Performance

Cible du cahier des charges : **inférence < 2 s par image sur CPU**. Mesures reproductibles :
```bash
python -m ml.evaluation.benchmark --runs 20      # -> docs/results/benchmark.md / .json
```

## Conditions de mesure (à lire avant les chiffres)
- CPU 20 threads, Windows, TensorFlow 2.21, **sans GPU**.
- **Machine non dédiée et charge non contrôlée** : d'autres applications tournaient sur ce poste (conteneurs Docker d'autres projets, autres serveurs locaux). *Correction :* une première version de ce document affirmait qu'un entraînement de ce projet tournait pendant les mesures ; c'était une hypothèse non vérifiée, et cet entraînement était en réalité déjà arrêté. La charge réelle pendant les mesures est donc inconnue : les chiffres sont bruités (par exemple ResNet50 « modèle seul » 353 ms > `predict()` complet 259 ms : même ordre de grandeur, écart dû au bruit, pas à la physique).
- Poids aléatoires (`weights=None`) : la latence dépend de l'architecture, pas des poids. Les photos de test sont synthétiques (JPEG 256 px et 3000×2250 px).
- 20 mesures par cas après échauffement ; on donne la médiane et le p95.

## Latence d'une prédiction (octets d'image → résultat)
| Modèle | Paramètres | Modèle seul | `predict()` photo 256 px | `predict()` photo 3000 px (p95) |
|---|---:|---:|---:|---:|
| baseline | 0,4 M | 34 ms | 42 ms | 105 ms (142) |
| MobileNetV2 | 2,3 M | 182 ms | 193 ms | 250 ms (314) |
| ResNet50 | 23,7 M | 353 ms | 259 ms | 365 ms (462) |
| VGG16 | 14,7 M | 114 ms | 132 ms | 212 ms (241) |

→ **Les 4 architectures respectent la cible de 2 s avec une marge d'un facteur 5 à 20**, sur une machine non dédiée.
Le choix du modèle final se fera donc sur la qualité (Phase 8), pas sur la vitesse. Observation
inattendue : MobileNetV2, réputé « léger », est plus lent que VGG16 ici (convolutions séparables peu
efficaces sur ce CPU/oneDNN) : ne pas supposer, mesurer.

## Où part le temps (photo 1024×768, côté serveur)
Validation 3 ms · second décodage + redimensionnement 10 ms · enregistrement fichier + SQLite 20 ms
→ l'application ajoute **~35 ms**. Le modèle domine ; le décodage d'une très grande photo (3000 px) coûte 70-90 ms.

## Débit avec plusieurs clients (HTTP, MobileNetV2, photo 1024×768)
| Clients simultanés | Débit | Latence médiane | p95 |
|---:|---:|---:|---:|
| 1 | 1,6 req/s | 625 ms | 691 ms |
| 4 | 3,6 req/s | 992 ms | 1 592 ms |
| 8 | 4,1 req/s | 1 638 ms | **2 398 ms** |

Le débit sature autour de 4 req/s : l'inférence est sérialisée par un verrou (le modèle n'est pas
thread-safe), les requêtes **font la queue**. À 8 clients simultanés, le p95 dépasse 2 s.
La latence d'1 client (625 ms) est supérieure à `predict()` (~250 ms) : l'écart est dû au client HTTP de
test et à la charge de la machine, pas au code applicatif (profilé : ~35 ms).

## Optimisations étudiées
| Idée | Résultat | Décision |
|---|---|---|
| Décodage JPEG réduit (`Image.draft`) | 74 → 14 ms pour une photo 3000 px, mais pixels légèrement modifiés (écart moyen 1,9/255) | **Non adoptée** : gain de ~60 ms sur un total de 250-450 ms, et effet non validé sur un modèle entraîné |
| Éviter le double décodage (validation + prédiction) | coût réel : 3 ms | Non adoptée : négligeable |
| `model(x)` plutôt que `model.predict(x)` | déjà utilisé (évite le surcoût de `predict`) | En place |
| Échauffement du modèle au démarrage | déjà fait (la 1ʳᵉ inférence coûte beaucoup plus) | En place |

## Écart d'entraînement / inférence mesuré
L'entraînement redimensionne avec TensorFlow (bilinéaire sans antialiasing), l'API avec Pillow
(bilinéaire antialiasé). Sur 20 vraies images du split test : écart moyen **1,0 / 255** par pixel
(max 30 sur les contours). Faible ; Pillow reste préférable pour les grandes photos de téléphone,
où le redimensionnement TF créerait de l'aliasing. **À confirmer en Phase 8** : comparer les
prédictions du modèle entraîné avec les deux chaînes sur le jeu de test.

## Pistes si la charge augmente (non implémentées)
- Plusieurs workers uvicorn (un modèle par processus : débit ×N, mémoire ×N) ; la limitation de débit en mémoire (Phase 17) devrait alors passer au niveau du proxy.
- Réduire le modèle (MobileNetV2 quantifié, TFLite) si le déploiement vise un petit serveur.
- File de requêtes avec réponse asynchrone si l'on veut absorber des pics sans dépasser 2 s.

## Limites
Aucune mesure sur GPU ni sur le serveur de production visé ; mesures faites sur une machine non dédiée
(charge inconnue) ; photos synthétiques. À refaire sur la machine cible, une fois libre.
