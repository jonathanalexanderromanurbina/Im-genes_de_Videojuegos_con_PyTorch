# 🎮 Clasificación de Imágenes de Videojuegos con PyTorch

Este proyecto implementa un sistema completo de clasificación de imágenes de videojuegos utilizando **Deep Learning** con PyTorch. El modelo es capaz de identificar diferentes géneros de videojuegos a partir de capturas de pantalla (screenshots).

## 🌟 Características

- ✅ **Múltiples arquitecturas**: CNN personalizada, ResNet50, EfficientNet
- ✅ **Data Augmentation**: Mejora la generalización del modelo
- ✅ **Transfer Learning**: Aprovecha modelos preentrenados
- ✅ **Descarga automática**: Integración con Kaggle API
- ✅ **Visualizaciones**: Gráficos de entrenamiento y matriz de confusión
- ✅ **Predicción por lotes**: Procesa múltiples imágenes
- ✅ **Notebook interactivo**: Jupyter notebook completo incluido

## 📁 Estructura del Proyecto

```
windsurf-project/
├── data/                       # Directorio para los datos
│   ├── raw/                   # Datos sin procesar
│   └── processed/             # Datos procesados
├── models/                    # Modelos guardados (.pth)
├── notebooks/                 # Jupyter notebooks
│   └── clasificacion_videojuegos.ipynb
├── src/                       # Código fuente
│   ├── data/                  # Módulo de datos
│   │   ├── __init__.py
│   │   └── dataset.py        # Carga y procesamiento de datos
│   ├── models/                # Módulo de modelos
│   │   ├── __init__.py
│   │   └── cnn_model.py      # Definiciones de modelos
│   ├── train.py              # Script de entrenamiento
│   ├── predict.py            # Script de predicción
│   └── __init__.py
├── quick_start.py            # Script de inicio rápido
├── requirements.txt          # Dependencias
├── .gitignore
└── README.md
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd windsurf-project
```

### 2. Crear un entorno virtual

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Kaggle API (Opcional)

Para descargar automáticamente el dataset desde Kaggle:

1. Crea una cuenta en [Kaggle](https://www.kaggle.com/)
2. Ve a `Account` → `API` → `Create New API Token`
3. Descarga el archivo `kaggle.json`
4. Colócalo en:
   - **Windows**: `C:\Users\<usuario>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

## 📊 Conjunto de Datos

Este proyecto utiliza el dataset **[gameplay-images](https://www.kaggle.com/datasets/aditmagotra/gameplay-images)** de Kaggle, que contiene imágenes de diferentes géneros de videojuegos.

El dataset se descarga automáticamente al ejecutar el entrenamiento si tienes configurada la API de Kaggle.

### Transformaciones de Datos

El proyecto implementa transformaciones robustas con data augmentation:

**Entrenamiento (con augmentation):**
- Resize a 224x224
- RandomRotation (±15°)
- RandomResizedCrop (scale 0.8-1.0)
- RandomHorizontalFlip (50%)
- Normalización con valores de ImageNet

**Validación/Test (sin augmentation):**
- Resize a 224x224
- Normalización con valores de ImageNet

La función `load_dataset_with_test()` garantiza que las transformaciones correctas se apliquen a cada split del dataset.

## 💻 Uso

### Opción 1: Inicio Rápido

La forma más sencilla de comenzar:

```bash
python quick_start.py
```

Este script:
- Descarga el dataset automáticamente
- Entrena un modelo CNN por 10 épocas
- Genera gráficos de resultados
- Guarda el modelo entrenado

### Opción 1b: Entrenamiento con Train/Val/Test Split

Para una evaluación más robusta con división 70/15/15:

```bash
python example_train_val_test.py
```

Este script:
- Divide el dataset en Train (70%), Validation (15%) y Test (15%)
- Aplica transformaciones correctas a cada split
- Evalúa en ambos conjuntos de validación y test
- Implementa data augmentation solo en entrenamiento

### Opción 2: Entrenamiento Personalizado

Para mayor control sobre los parámetros:

```bash
python src/train.py --data_dir data/raw \
                    --model cnn \
                    --epochs 15 \
                    --batch_size 32 \
                    --lr 0.001 \
                    --image_size 224
```

**Parámetros disponibles:**

- `--data_dir`: Directorio con el dataset
- `--kaggle_dataset`: Dataset de Kaggle (ej: `aditmagotra/gameplay-images`)
- `--model`: Tipo de modelo (`cnn`, `resnet`, `efficientnet`)
- `--batch_size`: Tamaño del batch (default: 32)
- `--epochs`: Número de épocas (default: 15)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Tamaño de imagen (default: 224)
- `--val_split`: Proporción de validación (default: 0.2)
- `--no_augment`: Desactivar data augmentation
- `--model_name`: Nombre del modelo a guardar

**Ejemplo con ResNet50:**

```bash
python src/train.py --kaggle_dataset aditmagotra/gameplay-images \
                    --model resnet \
                    --epochs 20 \
                    --batch_size 16
```

### Opción 3: Jupyter Notebook

Para un análisis interactivo:

```bash
jupyter notebook notebooks/clasificacion_videojuegos.ipynb
```

## 🔮 Predicción

### Predicción de una imagen individual

```bash
python src/predict.py --model_path models/videojuegos_classifier_final.pth \
                      --image_path ruta/a/imagen.jpg \
                      --save_output
```

### Predicción por lotes

```bash
python src/predict.py --model_path models/videojuegos_classifier_final.pth \
                      --image_dir ruta/a/directorio \
                      --save_output \
                      --output_dir predictions
```

**Parámetros:**

- `--model_path`: Ruta al modelo entrenado
- `--image_path`: Ruta a una imagen individual
- `--image_dir`: Directorio con múltiples imágenes
- `--save_output`: Guardar visualizaciones
- `--output_dir`: Directorio para guardar resultados

## 🏗️ Arquitecturas de Modelos

### 1. CNN Personalizada

Red convolucional diseñada específicamente para este problema:
- 4 bloques convolucionales con BatchNorm
- Adaptive pooling
- Dropout para regularización
- ~6M parámetros

### 2. ResNet50 (Transfer Learning)

- Modelo preentrenado en ImageNet
- Fine-tuning de todas las capas
- ~25M parámetros

### 3. EfficientNet-B0 (Transfer Learning)

- Arquitectura eficiente y moderna
- Preentrenado en ImageNet
- ~5M parámetros

## 📈 Resultados

Los modelos entrenados generan:

1. **Gráficos de entrenamiento**: Precisión y pérdida por época
2. **Matriz de confusión**: Visualización de predicciones
3. **Reporte de clasificación**: Precision, Recall, F1-Score por clase
4. **Historial JSON**: Métricas detalladas guardadas

Archivos generados en `models/`:
- `*_best.pth`: Mejor modelo durante entrenamiento
- `*_final.pth`: Modelo al finalizar
- `*_history.png`: Gráficos de entrenamiento
- `confusion_matrix.png`: Matriz de confusión
- `*_history.json`: Historial de métricas

## 🛠️ Tecnologías Utilizadas

- **PyTorch**: Framework de Deep Learning
- **torchvision**: Modelos y transformaciones
- **Kaggle API**: Descarga de datasets
- **scikit-learn**: Métricas de evaluación
- **matplotlib/seaborn**: Visualizaciones
- **Pillow**: Procesamiento de imágenes
- **tqdm**: Barras de progreso

## 📝 Ejemplos de Uso

### Entrenar con dataset de Kaggle (Train/Val)

```python
from src.data.dataset import download_kaggle_dataset, load_dataset
from src.models.cnn_model import get_model

# Descargar dataset
path = download_kaggle_dataset("aditmagotra/gameplay-images")

# Cargar datos (80% train, 20% val)
train_loader, val_loader, class_names = load_dataset(
    path, 
    batch_size=32,
    image_size=224,
    val_split=0.2,
    augment=True
)

# Crear modelo
model = get_model('cnn', num_classes=len(class_names))
```

### Entrenar con Train/Val/Test Split

```python
from src.data.dataset import load_dataset_with_test
from src.models.cnn_model import get_model

# Cargar datos con división 70/15/15
train_loader, val_loader, test_loader, class_names = load_dataset_with_test(
    data_dir='data/raw',
    batch_size=32,
    image_size=224,
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    augment=True,
    num_workers=0
)

# Crear modelo
model = get_model('cnn', num_classes=len(class_names))

# Entrenar...
# Evaluar en validación...
# Evaluar en test...
```

### Hacer predicciones

```python
from src.predict import load_trained_model, predict_image
from torchvision import transforms

# Cargar modelo
model, class_names, image_size = load_trained_model(
    'models/videojuegos_classifier_final.pth',
    device
)

# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predecir
prediction = predict_image(model, 'imagen.jpg', class_names, device, transform)
print(f"Clase: {prediction['class']} ({prediction['confidence']*100:.2f}%)")
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

Tu Nombre - [tu.email@example.com](mailto:tu.email@example.com)

## 🙏 Agradecimientos

- Dataset: [gameplay-images](https://www.kaggle.com/datasets/aditmagotra/gameplay-images) por Adit Magotra
- PyTorch Team por el excelente framework
- Comunidad de Kaggle por los datasets

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!
