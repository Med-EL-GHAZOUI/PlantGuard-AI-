"""Constantes partagées entre l'entraînement (Phases 5-8) et l'inférence (Phase 11).

Garder ces valeurs en un seul endroit évite un décalage entre ce que le modèle voit à
l'entraînement et ce qu'il voit en production (cause classique de mauvaises prédictions).
"""
IMG_SIZE = (224, 224)       # taille d'entrée attendue par VGG16 / ResNet50 / MobileNetV2
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES_FILE = "class_names.json"
