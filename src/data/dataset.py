"""
Utilidades para cargar y procesar datos de imágenes de videojuegos.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import kagglehub


def download_kaggle_dataset(dataset_path):
    """
    Descarga un dataset desde Kaggle usando kagglehub.
    
    Args:
        dataset_path (str): Ruta del dataset en Kaggle (ej: 'usuario/nombre-dataset')
    
    Returns:
        str: Ruta local donde se descargó el dataset
    """
    print(f"Descargando dataset desde Kaggle: {dataset_path}")
    path = kagglehub.dataset_download(dataset_path)
    print(f"Dataset descargado en: {path}")
    return path


def get_data_transforms(image_size=224, augment=True):
    """
    Obtiene las transformaciones para el dataset.
    
    Args:
        image_size (int): Tamaño de la imagen de salida
        augment (bool): Si se aplican aumentaciones de datos
    
    Returns:
        dict: Diccionario con transformaciones para train, val y test
    """
    if augment:
        # Transformaciones para entrenamiento (incluye Data Augmentation)
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Valores de ImageNet
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Transformaciones para validación y test (sin augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transforms, 
        'val': val_test_transforms,
        'test': val_test_transforms
    }


def load_dataset(data_dir, image_size=224, batch_size=32, val_split=0.2, augment=True, num_workers=0):
    """
    Carga el dataset de imágenes y crea los DataLoaders (train/val).
    
    Args:
        data_dir (str): Directorio raíz del dataset
        image_size (int): Tamaño de las imágenes
        batch_size (int): Tamaño del batch
        val_split (float): Proporción del dataset para validación
        augment (bool): Si se aplican aumentaciones de datos
        num_workers (int): Número de workers para cargar datos
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Obtener transformaciones
    data_transforms = get_data_transforms(image_size, augment)
    
    # Cargar dataset completo
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    
    # Dividir en train y validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Aplicar transformaciones
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Dataset cargado:")
    print(f"  - Total de imágenes: {dataset_size}")
    print(f"  - Imágenes de entrenamiento: {train_size}")
    print(f"  - Imágenes de validación: {val_size}")
    print(f"  - Número de clases: {len(class_names)}")
    print(f"  - Clases: {class_names}")
    
    return train_loader, val_loader, class_names


def load_dataset_with_test(data_dir, image_size=224, batch_size=32, 
                           train_split=0.70, val_split=0.15, test_split=0.15,
                           augment=True, num_workers=0):
    """
    Carga el dataset de imágenes y crea los DataLoaders (train/val/test).
    Implementación robusta que asegura transformaciones correctas para cada split.
    
    Args:
        data_dir (str): Directorio raíz del dataset
        image_size (int): Tamaño de las imágenes
        batch_size (int): Tamaño del batch
        train_split (float): Proporción del dataset para entrenamiento (default: 0.70)
        val_split (float): Proporción del dataset para validación (default: 0.15)
        test_split (float): Proporción del dataset para test (default: 0.15)
        augment (bool): Si se aplican aumentaciones de datos al entrenamiento
        num_workers (int): Número de workers para cargar datos
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    from torch.utils.data import Subset
    
    # Obtener transformaciones
    data_transforms = get_data_transforms(image_size, augment)
    
    # Crear el dataset base con transformaciones de entrenamiento
    full_dataset_train = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    class_names = full_dataset_train.classes
    
    # Crear datasets separados para val y test con transformaciones sin augmentation
    full_dataset_val = datasets.ImageFolder(data_dir, transform=data_transforms['val'])
    full_dataset_test = datasets.ImageFolder(data_dir, transform=data_transforms['test'])
    
    # Calcular tamaños de división
    dataset_size = len(full_dataset_train)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"\nTamaños de datasets: Train={train_size}, Val={val_size}, Test={test_size}, Total={dataset_size}")
    
    # Dividir el dataset usando random_split para obtener los índices
    train_dataset_temp, val_dataset_temp, test_dataset_temp = random_split(
        full_dataset_train, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Obtener los índices de cada split
    train_indices = train_dataset_temp.indices
    val_indices = val_dataset_temp.indices
    test_indices = test_dataset_temp.indices
    
    # Crear subsets con las transformaciones correctas
    train_dataset = Subset(full_dataset_train, train_indices)  # Conserva transform_train
    val_dataset = Subset(full_dataset_val, val_indices)        # Usa transform_val
    test_dataset = Subset(full_dataset_test, test_indices)     # Usa transform_test
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("DataLoaders creados exitosamente.")
    print(f"  - Número de clases: {len(class_names)}")
    print(f"  - Clases: {class_names}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Data augmentation: {'Activado' if augment else 'Desactivado'}")
    
    return train_loader, val_loader, test_loader, class_names


class CustomImageDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes desde un directorio.
    """
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directorio con las imágenes
            transform (callable, optional): Transformaciones a aplicar
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.images[idx]


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Desnormaliza una imagen tensor para visualización.
    
    Args:
        tensor (torch.Tensor): Imagen normalizada
        mean (list): Media usada en la normalización
        std (list): Desviación estándar usada en la normalización
    
    Returns:
        torch.Tensor: Imagen desnormalizada
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
