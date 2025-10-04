"""
Módulo de modelos para clasificación de imágenes de videojuegos.
"""

from .cnn_model import VideogameCNN, ResNetTransferLearning, EfficientNetTransferLearning, get_model

__all__ = ['VideogameCNN', 'ResNetTransferLearning', 'EfficientNetTransferLearning', 'get_model']
