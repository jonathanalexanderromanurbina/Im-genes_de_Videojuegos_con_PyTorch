"""
Script de entrenamiento para clasificación de imágenes de videojuegos con PyTorch.
"""

import os
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from models.cnn_model import get_model
from data.dataset import load_dataset, download_kaggle_dataset

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                device, num_epochs=25, model_dir='models', model_name='videojuegos_model'):
    """
    Entrena el modelo.
    
    Args:
        model: Modelo a entrenar
        criterion: Función de pérdida
        optimizer: Optimizador
        scheduler: Planificador de tasa de aprendizaje
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        device: Dispositivo (CPU o GPU)
        num_epochs: Número de épocas
        model_dir: Directorio para guardar modelos
        model_name: Nombre base del modelo
        
    Returns:
        tuple: (model, history)
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Entrenamiento'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        
        # Fase de validación
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validación'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Guardar el mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Guardar checkpoint
            checkpoint_path = os.path.join(model_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'✓ Mejor modelo guardado con accuracy: {val_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'\n{"="*60}')
    print(f'Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Mejor precisión en validación: {best_acc:.4f}')
    print(f'{"="*60}')
    
    # Cargar los pesos del mejor modelo
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(history, save_path='models/training_history.png'):
    """Grafica la precisión y pérdida del entrenamiento y validación."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de precisión
    axes[0].plot(history['train_acc'], label='Entrenamiento', marker='o', linewidth=2)
    if 'val_acc' in history:
        axes[0].plot(history['val_acc'], label='Validación', marker='s', linewidth=2)
    axes[0].set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Precisión', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de pérdida
    axes[1].plot(history['train_loss'], label='Entrenamiento', marker='o', linewidth=2)
    if 'val_loss' in history:
        axes[1].plot(history['val_loss'], label='Validación', marker='s', linewidth=2)
    axes[1].set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Pérdida', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Gráfico guardado en: {save_path}')
    plt.show()


def evaluate_model(model, val_loader, device, class_names):
    """
    Evalúa el modelo y genera métricas detalladas.
    
    Args:
        model: Modelo entrenado
        val_loader: DataLoader de validación
        device: Dispositivo (CPU o GPU)
        class_names: Lista de nombres de clases
    
    Returns:
        dict: Diccionario con métricas de evaluación
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluando modelo...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluación'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f'\n{"="*60}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'{"="*60}')
    
    # Reporte de clasificación
    print('\nReporte de Clasificación:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Número de predicciones'})
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Verdadero', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print('Matriz de confusión guardada en: models/confusion_matrix.png')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }

def main():
    """Función principal para entrenar el modelo."""
    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Entrenamiento de clasificador de imágenes de videojuegos')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Directorio de datos')
    parser.add_argument('--kaggle_dataset', type=str, default=None, help='Dataset de Kaggle (ej: usuario/dataset)')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet', 'efficientnet'],
                       help='Tipo de modelo a usar')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Tamaño de imagen')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proporción de validación')
    parser.add_argument('--no_augment', action='store_true', help='Desactivar data augmentation')
    parser.add_argument('--model_name', type=str, default='videojuegos_classifier', help='Nombre del modelo')
    
    args = parser.parse_args()
    
    # Configuración de device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Usando dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Crear directorio para modelos
    os.makedirs('models', exist_ok=True)
    
    # Descargar dataset de Kaggle si se especifica
    if args.kaggle_dataset:
        print(f"Descargando dataset de Kaggle: {args.kaggle_dataset}")
        data_path = download_kaggle_dataset(args.kaggle_dataset)
        args.data_dir = data_path
    
    # Cargar datos
    print(f"\nCargando datos desde: {args.data_dir}")
    train_loader, val_loader, class_names = load_dataset(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        augment=not args.no_augment,
        num_workers=0  # Cambiar a 4 si tienes múltiples cores
    )
    
    num_classes = len(class_names)
    print(f"\nNúmero de clases: {num_classes}")
    print(f"Clases: {class_names}\n")
    
    # Crear modelo
    print(f"Creando modelo: {args.model}")
    if args.model == 'cnn':
        model = get_model('cnn', num_classes=num_classes)
    elif args.model == 'resnet':
        model = get_model('resnet', num_classes=num_classes, pretrained=True, freeze_layers=False)
    else:  # efficientnet
        model = get_model('efficientnet', num_classes=num_classes, pretrained=True, freeze_layers=False)
    
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")
    
    # Definir loss, optimizer y scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Entrenar modelo
    print("Iniciando entrenamiento...\n")
    model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        model_dir='models',
        model_name=args.model_name
    )
    
    # Guardar historial de entrenamiento
    import json
    history_path = os.path.join('models', f'{args.model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f'\nHistorial guardado en: {history_path}')
    
    # Graficar historial
    plot_training_history(history, save_path=f'models/{args.model_name}_history.png')
    
    # Evaluar modelo
    metrics = evaluate_model(model, val_loader, device, class_names)
    
    # Guardar modelo final completo
    final_model_path = os.path.join('models', f'{args.model_name}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'model_type': args.model,
        'accuracy': metrics['accuracy'],
        'image_size': args.image_size
    }, final_model_path)
    print(f'\nModelo final guardado en: {final_model_path}')
    
    print(f"\n{'='*60}")
    print("¡Entrenamiento completado exitosamente!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
