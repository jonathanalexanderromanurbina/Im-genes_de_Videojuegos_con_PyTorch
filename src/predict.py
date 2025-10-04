"""
Script de predicción/inferencia para clasificación de imágenes de videojuegos.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.cnn_model import get_model
from data.dataset import get_data_transforms

def predict_image(model, image_path, class_names, device, transform=None):
    """
    Realiza una predicción sobre una única imagen.
    
    Args:
        model: Modelo entrenado.
        image_path: Ruta a la imagen.
        class_names: Lista de nombres de las clases.
        device: Dispositivo donde se ejecutará el modelo.
        transform: Transformaciones a aplicar a la imagen.
        
    Returns:
        Predicción como un diccionario con la clase predicha y las probabilidades.
    """
    # Cargar y transformar la imagen
    image = Image.open(image_path).convert('RGB')
    
    if transform is not None:
        image = transform(image).unsqueeze(0).to(device)
    
    # Poner el modelo en modo evaluación
    model.eval()
    
    # Realizar la predicción
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Obtener las 5 clases con mayor probabilidad
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Convertir a listas
    top5_prob = top5_prob.cpu().numpy()
    top5_catid = top5_catid.cpu().numpy()
    
    # Crear diccionario con los resultados
    result = {
        'class': class_names[top5_catid[0]],
        'class_id': int(top5_catid[0]),
        'confidence': float(top5_prob[0]),
        'top5': [
            {'class': class_names[cat_id], 'probability': float(prob)}
            for prob, cat_id in zip(top5_prob, top5_catid)
        ]
    }
    
    return result

def show_prediction(image_path, prediction, save_path=None):
    """Muestra la imagen con la predicción."""
    # Cargar la imagen
    image = Image.open(image_path)
    
    # Crear la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mostrar la imagen
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicción: {prediction["class"]}\nConfianza: {prediction["confidence"]*100:.2f}%',
                  fontsize=12, fontweight='bold')
    
    # Mostrar las predicciones
    y_pos = np.arange(len(prediction['top5']))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(prediction['top5'])))
    bars = ax2.barh(y_pos, [p['probability'] for p in prediction['top5']], 
                    align='center', color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([p['class'] for p in prediction['top5']])
    ax2.set_xlabel('Probabilidad', fontsize=11)
    ax2.set_title('Top 5 Predicciones', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim([0, 1])
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width*100:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Predicción guardada en: {save_path}')
    
    plt.show()


def load_trained_model(model_path, device):
    """
    Carga un modelo entrenado desde un checkpoint.
    
    Args:
        model_path (str): Ruta al archivo del modelo
        device: Dispositivo (CPU o GPU)
    
    Returns:
        tuple: (model, class_names, image_size)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    print(f"Cargando modelo desde: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraer información del checkpoint
    if 'model_state_dict' in checkpoint:
        # Formato nuevo
        model_state = checkpoint['model_state_dict']
        class_names = checkpoint.get('class_names', [])
        num_classes = checkpoint.get('num_classes', len(class_names))
        model_type = checkpoint.get('model_type', 'cnn')
        image_size = checkpoint.get('image_size', 224)
        accuracy = checkpoint.get('accuracy', None)
    else:
        # Formato antiguo (solo state_dict)
        model_state = checkpoint
        class_names = []
        num_classes = None
        model_type = 'cnn'
        image_size = 224
        accuracy = None
    
    # Inferir num_classes si no está disponible
    if num_classes is None:
        # Buscar la última capa para determinar num_classes
        for key in model_state.keys():
            if 'fc' in key and 'weight' in key:
                num_classes = model_state[key].shape[0]
                break
    
    # Crear modelo
    model = get_model(model_type, num_classes=num_classes)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    print(f"Modelo cargado exitosamente")
    print(f"  - Tipo: {model_type}")
    print(f"  - Clases: {num_classes}")
    if accuracy:
        print(f"  - Accuracy: {accuracy:.4f}")
    
    return model, class_names, image_size

def predict_batch(model, image_dir, class_names, device, transform, top_k=5):
    """
    Realiza predicciones sobre un directorio de imágenes.
    
    Args:
        model: Modelo entrenado
        image_dir (str): Directorio con imágenes
        class_names (list): Nombres de clases
        device: Dispositivo
        transform: Transformaciones
        top_k (int): Número de predicciones top a retornar
    
    Returns:
        list: Lista de predicciones
    """
    results = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # Obtener todas las imágenes
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    print(f"\nProcesando {len(image_files)} imágenes...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            prediction = predict_image(model, img_path, class_names, device, transform)
            results.append({
                'filename': img_file,
                'prediction': prediction
            })
            print(f"✓ {img_file}: {prediction['class']} ({prediction['confidence']*100:.1f}%)")
        except Exception as e:
            print(f"✗ Error procesando {img_file}: {str(e)}")
    
    return results


def main():
    """Función principal para realizar predicciones."""
    parser = argparse.ArgumentParser(description='Predicción de imágenes de videojuegos')
    parser.add_argument('--model_path', type=str, default='models/videojuegos_classifier_final.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Ruta a una imagen individual')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directorio con múltiples imágenes')
    parser.add_argument('--save_output', action='store_true',
                       help='Guardar visualizaciones de predicciones')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Verificar argumentos
    if not args.image_path and not args.image_dir:
        print("Error: Debes especificar --image_path o --image_dir")
        return
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Usando dispositivo: {device}")
    print(f"{'='*60}\n")
    
    # Cargar modelo
    try:
        model, class_names, image_size = load_trained_model(args.model_path, device)
    except Exception as e:
        print(f"Error cargando modelo: {str(e)}")
        return
    
    # Definir transformaciones
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Crear directorio de salida si es necesario
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Predicción de imagen individual
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: No se encontró la imagen en {args.image_path}")
            return
        
        print(f"Realizando predicción para: {args.image_path}\n")
        prediction = predict_image(model, args.image_path, class_names, device, transform)
        
        # Mostrar resultados
        print(f"\n{'='*60}")
        print(f"Predicción para: {os.path.basename(args.image_path)}")
        print(f"{'='*60}")
        print(f"Clase predicha: {prediction['class']}")
        print(f"Confianza: {prediction['confidence']*100:.2f}%")
        
        print(f"\nTop 5 predicciones:")
        for i, item in enumerate(prediction['top5'], 1):
            print(f"  {i}. {item['class']}: {item['probability']*100:.2f}%")
        
        # Visualizar
        save_path = None
        if args.save_output:
            save_path = os.path.join(args.output_dir, 
                                    f"pred_{os.path.basename(args.image_path)}")
        
        show_prediction(args.image_path, prediction, save_path)
    
    # Predicción de directorio
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"Error: No se encontró el directorio {args.image_dir}")
            return
        
        results = predict_batch(model, args.image_dir, class_names, device, transform)
        
        print(f"\n{'='*60}")
        print(f"Resumen de predicciones:")
        print(f"{'='*60}")
        print(f"Total de imágenes procesadas: {len(results)}")
        
        # Contar predicciones por clase
        class_counts = {}
        for result in results:
            pred_class = result['prediction']['class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print(f"\nDistribución de predicciones:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} imágenes")
        
        # Guardar resultados en archivo
        if args.save_output:
            import json
            output_file = os.path.join(args.output_dir, 'predictions.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"\nResultados guardados en: {output_file}")


if __name__ == '__main__':
    main()
