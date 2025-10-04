# Changelog

## [1.1.0] - 2025-10-04

### ✨ Nuevas Características

#### División Train/Val/Test Robusta
- **Nueva función**: `load_dataset_with_test()` en `src/data/dataset.py`
- Implementa división 70/15/15 (configurable) del dataset
- Garantiza que las transformaciones correctas se apliquen a cada split
- Usa `Subset` para mantener datasets separados con transformaciones independientes

#### Transformaciones Mejoradas
- **Data Augmentation actualizado**:
  - `RandomRotation(15)`: Rotación aleatoria ±15°
  - `RandomResizedCrop`: Crop aleatorio con escala 0.8-1.0
  - `RandomHorizontalFlip`: Flip horizontal con probabilidad 50%
  - Normalización con valores de ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

- **Transformaciones separadas**:
  - Train: Con data augmentation completo
  - Val/Test: Solo resize y normalización (sin augmentation)

#### Script de Ejemplo
- **Nuevo archivo**: `example_train_val_test.py`
- Demuestra el uso completo de la división train/val/test
- Evalúa el modelo en ambos conjuntos de validación y test
- Guarda métricas de accuracy para val y test

### 📚 Documentación

- Actualizado `README.md` con:
  - Sección de transformaciones de datos
  - Ejemplos de uso de `load_dataset_with_test()`
  - Opción 1b en la guía de uso
  - Ejemplos de código mejorados

- Actualizado `src/data/__init__.py`:
  - Exporta la nueva función `load_dataset_with_test`

### 🔧 Mejoras Técnicas

#### Implementación Robusta de Splits
```python
# Antes: Transformaciones compartidas (problema)
full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# Problema: val_dataset usa transform_train

# Ahora: Transformaciones independientes (correcto)
full_dataset_train = datasets.ImageFolder(data_dir, transform=transform_train)
full_dataset_val = datasets.ImageFolder(data_dir, transform=transform_val)
train_indices = train_dataset_temp.indices
val_indices = val_dataset_temp.indices
train_dataset = Subset(full_dataset_train, train_indices)
val_dataset = Subset(full_dataset_val, val_indices)
# Correcto: Cada split tiene sus propias transformaciones
```

### 📊 Estructura de Archivos

```
Nuevos archivos:
├── example_train_val_test.py    # Script de ejemplo train/val/test
└── CHANGELOG.md                  # Este archivo

Archivos modificados:
├── src/data/dataset.py           # Nueva función load_dataset_with_test()
├── src/data/__init__.py          # Exporta nueva función
└── README.md                     # Documentación actualizada
```

### 🎯 Beneficios

1. **Evaluación más robusta**: Separación clara entre validación y test
2. **Transformaciones correctas**: Data augmentation solo en entrenamiento
3. **Reproducibilidad**: Seed fijo (42) para splits consistentes
4. **Flexibilidad**: Proporciones de split configurables
5. **Mejores prácticas**: Implementación según estándares de ML

### 💡 Uso Recomendado

**Para experimentación rápida:**
```bash
python quick_start.py  # División train/val (80/20)
```

**Para evaluación rigurosa:**
```bash
python example_train_val_test.py  # División train/val/test (70/15/15)
```

**Para producción:**
```python
# Usar load_dataset_with_test() con splits personalizados
train_loader, val_loader, test_loader, class_names = load_dataset_with_test(
    data_dir='data/raw',
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    augment=True
)
```

---

## [1.0.0] - 2025-10-04

### 🎉 Lanzamiento Inicial

- Implementación completa de clasificación de imágenes de videojuegos
- 3 arquitecturas de modelos (CNN, ResNet50, EfficientNet)
- Scripts de entrenamiento y predicción
- Notebook interactivo de Jupyter
- Integración con Kaggle API
- Documentación completa
- Data augmentation básico
- Visualizaciones y métricas

---

**Formato basado en [Keep a Changelog](https://keepachangelog.com/)**
