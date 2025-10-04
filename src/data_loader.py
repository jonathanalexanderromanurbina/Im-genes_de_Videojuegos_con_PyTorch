import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

class VideoGameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directorio con las imágenes organizadas en subcarpetas por clase.
            transform (callable, optional): Transformaciones a aplicar a las imágenes.
        """
        self.data = datasets.ImageFolder(root=data_dir, transform=transform)
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """
    Crea los DataLoaders para entrenamiento y validación.
    
    Args:
        data_dir (str): Ruta al directorio raíz de los datos.
        batch_size (int): Tamaño del lote.
        img_size (int): Tamaño al que se redimensionarán las imágenes.
        
    Returns:
        train_loader, val_loader: DataLoaders para entrenamiento y validación.
        class_names: Lista con los nombres de las clases.
    """
    # Transformaciones para los datos de entrenamiento (con aumento de datos)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para los datos de validación (sin aumento de datos)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Crear conjuntos de datos
    train_dataset = VideoGameDataset(
        os.path.join(data_dir, 'train'), 
        transform=train_transform
    )
    
    val_dataset = VideoGameDataset(
        os.path.join(data_dir, 'test'),  # O 'val' si tienes un conjunto de validación separado
        transform=val_transform
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, train_dataset.classes

def imshow(inp, title=None):
    """Mostrar imagen para Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pausa para actualizar los gráficos
