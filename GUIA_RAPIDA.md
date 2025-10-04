# 🚀 Guía Rápida - Clasificación de Imágenes de Videojuegos

## ⚡ Inicio en 3 Pasos

### 1️⃣ Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2️⃣ Configurar Kaggle (Opcional)

Coloca tu `kaggle.json` en:
- **Windows**: `C:\Users\<usuario>\.kaggle\kaggle.json`
- **Linux/Mac**: `~/.kaggle/kaggle.json`

### 3️⃣ Entrenar el Modelo

```bash
python quick_start.py
```

¡Eso es todo! El script descargará el dataset y entrenará el modelo automáticamente.

---

## 📋 Comandos Útiles

### Entrenamiento Básico

```bash
# Con dataset de Kaggle (descarga automática)
python src/train.py --kaggle_dataset aditmagotra/gameplay-images --epochs 15

# Con dataset local
python src/train.py --data_dir data/raw --epochs 15
```

### Entrenamiento con ResNet50

```bash
python src/train.py --kaggle_dataset aditmagotra/gameplay-images \
                    --model resnet \
                    --epochs 20 \
                    --batch_size 16
```

### Predicción de una Imagen

```bash
python src/predict.py --model_path models/videojuegos_classifier_final.pth \
                      --image_path mi_imagen.jpg \
                      --save_output
```

### Predicción de Múltiples Imágenes

```bash
python src/predict.py --model_path models/videojuegos_classifier_final.pth \
                      --image_dir carpeta_imagenes/ \
                      --save_output
```

---

## 🎯 Modelos Disponibles

| Modelo | Comando | Parámetros | Velocidad |
|--------|---------|------------|-----------|
| **CNN Personalizada** | `--model cnn` | ~6M | ⚡⚡⚡ Rápido |
| **ResNet50** | `--model resnet` | ~25M | ⚡⚡ Medio |
| **EfficientNet-B0** | `--model efficientnet` | ~5M | ⚡⚡⚡ Rápido |

---

## 📊 Estructura de Archivos Generados

Después del entrenamiento, encontrarás:

```
models/
├── videojuegos_classifier_best.pth      # Mejor modelo
├── videojuegos_classifier_final.pth     # Modelo final
├── videojuegos_classifier_history.png   # Gráficos
├── videojuegos_classifier_history.json  # Métricas
└── confusion_matrix.png                 # Matriz de confusión
```

---

## 🐛 Solución de Problemas

### Error: "No module named 'kagglehub'"

```bash
pip install kagglehub
```

### Error: "Kaggle API credentials not found"

1. Ve a https://www.kaggle.com/
2. Account → API → Create New API Token
3. Descarga `kaggle.json`
4. Colócalo en `~/.kaggle/` (Linux/Mac) o `C:\Users\<usuario>\.kaggle\` (Windows)

### Error: "CUDA out of memory"

Reduce el batch size:

```bash
python src/train.py --batch_size 16  # o incluso 8
```

### El entrenamiento es muy lento

Si no tienes GPU, reduce épocas y batch size:

```bash
python src/train.py --epochs 5 --batch_size 16
```

---

## 💡 Tips

### 1. Monitorear GPU (si tienes NVIDIA)

```bash
# En otra terminal
nvidia-smi -l 1
```

### 2. Usar Jupyter Notebook

```bash
jupyter notebook notebooks/clasificacion_videojuegos.ipynb
```

### 3. Guardar Logs

```bash
python src/train.py --epochs 15 2>&1 | tee training.log
```

### 4. Continuar Entrenamiento

Modifica `src/train.py` para cargar un checkpoint existente.

---

## 📈 Mejores Prácticas

✅ **Data Augmentation**: Activado por defecto (mejora generalización)  
✅ **Early Stopping**: El mejor modelo se guarda automáticamente  
✅ **Learning Rate Scheduling**: Reduce LR cada 7 épocas  
✅ **Batch Normalization**: Estabiliza el entrenamiento  
✅ **Dropout**: Previene overfitting  

---

## 🎓 Recursos Adicionales

- **Documentación PyTorch**: https://pytorch.org/docs/
- **Dataset Original**: https://www.kaggle.com/datasets/aditmagotra/gameplay-images
- **Transfer Learning**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

---

## 📞 Soporte

¿Problemas? Abre un issue en el repositorio o contacta al autor.

---

**¡Feliz entrenamiento! 🎮🤖**
