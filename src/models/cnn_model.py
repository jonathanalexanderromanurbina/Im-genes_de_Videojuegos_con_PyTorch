"""
Modelos de redes neuronales convolucionales para clasificación de imágenes de videojuegos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VideogameCNN(nn.Module):
    """
    Red neuronal convolucional personalizada para clasificación de imágenes de videojuegos.
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(VideogameCNN, self).__init__()
        
        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque convolucional 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloque convolucional 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bloque convolucional 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Capas fully connected
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Bloque 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Bloque 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Bloque 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Bloque 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        # Adaptive pooling y flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Capas fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class ResNetTransferLearning(nn.Module):
    """
    Modelo basado en ResNet50 con transfer learning.
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_layers=True):
        super(ResNetTransferLearning, self).__init__()
        
        # Cargar ResNet50 preentrenado
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Congelar capas si se especifica
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Reemplazar la última capa fully connected
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)


class EfficientNetTransferLearning(nn.Module):
    """
    Modelo basado en EfficientNet con transfer learning.
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_layers=True):
        super(EfficientNetTransferLearning, self).__init__()
        
        # Cargar EfficientNet-B0 preentrenado
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Congelar capas si se especifica
        if freeze_layers:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
        
        # Reemplazar el clasificador
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_name='cnn', num_classes=10, **kwargs):
    """
    Función auxiliar para obtener un modelo por nombre.
    
    Args:
        model_name (str): Nombre del modelo ('cnn', 'resnet', 'efficientnet')
        num_classes (int): Número de clases a clasificar
        **kwargs: Argumentos adicionales para el modelo
    
    Returns:
        nn.Module: Modelo de PyTorch
    """
    models_dict = {
        'cnn': VideogameCNN,
        'resnet': ResNetTransferLearning,
        'efficientnet': EfficientNetTransferLearning
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Modelo '{model_name}' no reconocido. Opciones: {list(models_dict.keys())}")
    
    return models_dict[model_name](num_classes=num_classes, **kwargs)
