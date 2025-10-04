"""
Script de inicio rápido para entrenar el clasificador de imágenes de videojuegos.

Uso:
    python quick_start.py
"""

import os
import sys

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from models.cnn_model import get_model
from data.dataset import download_kaggle_dataset, load_dataset

def main():
    """Función principal de inicio rápido."""
    
    print("="*60)
    print("CLASIFICACIÓN DE IMÁGENES DE VIDEOJUEGOS")
    print("="*60)
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Parámetros
    KAGGLE_DATASET = "aditmagotra/gameplay-images"
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    
    print(f"\nParámetros:")
    print(f"  - Dataset: {KAGGLE_DATASET}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Épocas: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Tamaño de imagen: {IMAGE_SIZE}")
    
    # Descargar dataset
    print(f"\n{'='*60}")
    print("DESCARGANDO DATASET")
    print("="*60)
    
    try:
        path = download_kaggle_dataset(KAGGLE_DATASET)
        
        # Buscar la carpeta correcta
        data_dir = os.path.join(path, 'gameplay-images')
        if not os.path.exists(data_dir):
            data_dir = path
        
        print(f"Dataset descargado en: {data_dir}")
        
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        print("Intentando usar directorio local...")
        data_dir = 'data/raw'
        
        if not os.path.exists(data_dir):
            print("\nERROR: No se encontró el dataset.")
            print("Por favor:")
            print("1. Configura tu API key de Kaggle en ~/.kaggle/kaggle.json")
            print("2. O descarga manualmente el dataset y colócalo en data/raw/")
            return
    
    # Verificar clases
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    
    print(f"\nClases encontradas ({len(classes)}):")
    for i, cls in enumerate(classes, 1):
        class_path = os.path.join(data_dir, cls)
        count = len(os.listdir(class_path))
        print(f"  {i}. {cls}: {count} imágenes")
    
    # Cargar datos
    print(f"\n{'='*60}")
    print("CARGANDO DATOS")
    print("="*60)
    
    train_loader, val_loader, class_names = load_dataset(
        data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        augment=True,
        num_workers=0
    )
    
    num_classes = len(class_names)
    
    # Crear modelo
    print(f"\n{'='*60}")
    print("CREANDO MODELO")
    print("="*60)
    
    model = get_model('cnn', num_classes=num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModelo: CNN Personalizado")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Entrenar
    print(f"\n{'='*60}")
    print("ENTRENANDO MODELO")
    print("="*60)
    print(f"\nIniciando entrenamiento por {EPOCHS} épocas...\n")
    
    # Importar función de entrenamiento
    from train import train_model, plot_training_history, evaluate_model
    
    os.makedirs('models', exist_ok=True)
    
    model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=EPOCHS,
        model_dir='models',
        model_name='videojuegos_quick_start'
    )
    
    # Graficar resultados
    print(f"\n{'='*60}")
    print("GENERANDO GRÁFICOS")
    print("="*60)
    
    plot_training_history(history, save_path='models/quick_start_history.png')
    
    # Evaluar
    print(f"\n{'='*60}")
    print("EVALUANDO MODELO")
    print("="*60)
    
    metrics = evaluate_model(model, val_loader, device, class_names)
    
    # Guardar modelo final
    final_model_path = 'models/videojuegos_quick_start_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'model_type': 'cnn',
        'accuracy': metrics['accuracy'],
        'image_size': IMAGE_SIZE
    }, final_model_path)
    
    print(f"\n{'='*60}")
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    print(f"\nModelo guardado en: {final_model_path}")
    print(f"Accuracy final: {metrics['accuracy']:.4f}")
    print(f"\nPara hacer predicciones, ejecuta:")
    print(f"  python src/predict.py --model_path {final_model_path} --image_path <ruta_imagen>")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
