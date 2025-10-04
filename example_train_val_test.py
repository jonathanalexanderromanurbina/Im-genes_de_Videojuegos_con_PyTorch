"""
Script de ejemplo para entrenar con división Train/Val/Test (70/15/15).

Este script demuestra el uso de la función load_dataset_with_test que implementa
una división robusta del dataset con transformaciones correctas para cada split.

Uso:
    python example_train_val_test.py
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
from data.dataset import download_kaggle_dataset, load_dataset_with_test


def main():
    """Función principal con división train/val/test."""
    
    print("="*60)
    print("CLASIFICACIÓN DE VIDEOJUEGOS - TRAIN/VAL/TEST SPLIT")
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
    
    # División del dataset: 70% Train, 15% Val, 15% Test
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    print(f"\nParámetros:")
    print(f"  - Dataset: {KAGGLE_DATASET}")
    print(f"  - División: Train={TRAIN_SPLIT*100:.0f}%, Val={VAL_SPLIT*100:.0f}%, Test={TEST_SPLIT*100:.0f}%")
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
    
    # Cargar datos con división train/val/test
    print(f"\n{'='*60}")
    print("CARGANDO DATOS (TRAIN/VAL/TEST)")
    print("="*60)
    
    train_loader, val_loader, test_loader, class_names = load_dataset_with_test(
        data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        augment=True,
        num_workers=0
    )
    
    num_classes = len(class_names)
    
    print(f"\n✓ Datos cargados correctamente")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
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
        model_name='videojuegos_train_val_test'
    )
    
    # Graficar resultados
    print(f"\n{'='*60}")
    print("GENERANDO GRÁFICOS")
    print("="*60)
    
    plot_training_history(history, save_path='models/train_val_test_history.png')
    
    # Evaluar en conjunto de validación
    print(f"\n{'='*60}")
    print("EVALUANDO EN CONJUNTO DE VALIDACIÓN")
    print("="*60)
    
    val_metrics = evaluate_model(model, val_loader, device, class_names)
    
    # Evaluar en conjunto de test
    print(f"\n{'='*60}")
    print("EVALUANDO EN CONJUNTO DE TEST")
    print("="*60)
    
    test_metrics = evaluate_model(model, test_loader, device, class_names)
    
    # Guardar modelo final
    final_model_path = 'models/videojuegos_train_val_test_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'model_type': 'cnn',
        'val_accuracy': val_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'image_size': IMAGE_SIZE
    }, final_model_path)
    
    print(f"\n{'='*60}")
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    print(f"\nModelo guardado en: {final_model_path}")
    print(f"Accuracy en Validación: {val_metrics['accuracy']:.4f}")
    print(f"Accuracy en Test: {test_metrics['accuracy']:.4f}")
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
