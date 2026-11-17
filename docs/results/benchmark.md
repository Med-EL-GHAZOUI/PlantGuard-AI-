| Modèle | Paramètres | Prétraitement 256 px | Prétraitement 3000 px | Modèle seul | predict() photo 256 px | predict() photo 3000 px | < 2 s ? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| baseline | 399,622 | 1 ms | 91 ms | 34 ms | 42 ms | 105 ms (p95 142) | oui |
| mobilenetv2 | 2,306,674 | 1 ms | 70 ms | 182 ms | 193 ms | 250 ms (p95 314) | oui |
| resnet50 | 23,665,586 | 1 ms | 78 ms | 353 ms | 259 ms | 365 ms (p95 462) | oui |
| vgg16 | 14,734,194 | 1 ms | 72 ms | 114 ms | 132 ms | 212 ms (p95 241) | oui |
