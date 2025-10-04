import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import lr_scheduler

def create_model(num_classes, use_pretrained=True):
    """
    Crea un modelo de red neuronal convolucional para clasificación de imágenes.
    
    Args:
        num_classes (int): Número de clases de salida.
        use_pretrained (bool): Si es True, usa pesos pre-entrenados en ImageNet.
        
    Returns:
        model: Modelo de PyTorch.
        criterion: Función de pérdida.
        optimizer: Optimizador.
        scheduler: Planificador de tasa de aprendizaje.
    """
    # Cargar un modelo pre-entrenado (ResNet18 en este caso)
    model = models.resnet18(pretrained=use_pretrained)
    
    # Congelar los parámetros de las capas convolucionales
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar la capa completamente conectada final
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    # Función de pérdida
    criterion = nn.CrossEntropyLoss()
    
    # Solo los parámetros de la última capa requieren gradientes
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Planificador de tasa de aprendizaje
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return model, criterion, optimizer, scheduler

def save_model(model, path, class_names, optimizer=None, epoch=None, val_loss=None):
    """
    Guarda el modelo entrenado.
    
    Args:
        model: Modelo a guardar.
        path: Ruta donde se guardará el modelo.
        class_names: Lista de nombres de las clases.
        optimizer: Optimizador (opcional).
        epoch: Época actual (opcional).
        val_loss: Pérdida de validación (opcional).
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'val_loss': val_loss,
        'class_names': class_names
    }, path)

def load_model(path, num_classes, use_pretrained=True):
    """
    Carga un modelo guardado.
    
    Args:
        path: Ruta al modelo guardado.
        num_classes: Número de clases.
        use_pretrained: Si es True, carga pesos pre-entrenados.
        
    Returns:
        model: Modelo cargado.
        class_names: Lista de nombres de las clases.
    """
    checkpoint = torch.load(path)
    model, _, _, _ = create_model(num_classes, use_pretrained)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint.get('class_names', [])
    return model, class_names
