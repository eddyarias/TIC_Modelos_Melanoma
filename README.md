# TIC_Modelos_Melanoma
En este repositorio contiene los scripts de preprocesamiento, entrenamiento y evaluación de los modelos CNN y Transformer

## Estructura del proyecto

**directorio baseCode**: contiene el código básico para entrenamientos de redes neuronales y transformers
- `utils/csv_dataset_builder.py`: Utilidad para construir datasets desde archivos CSV
- `utils/`: Módulos auxiliares (training, tensorboard, manual_stop, etc.)
- `models/`: Definiciones de modelos (classification, siamese, etc.)
- `dataloaders/`: Cargadores de datos y data augmentation

**directorio database**: contiene las etiquetas y los datos para entrenamiento, prueba y validación

## Uso básico

# ejecutar un entrenamiento
python baseCode/train_classification.py \
  -vs 0.1 \
  -ts 0.05 \
  -lim 20000 \
  -b convnext_large \
  -e 25 \
  -bs 16 \
  -w imagenet \
  -lvl heavy \
  -lr 0.001 \
  -tb

# comenzar desde un checkpoint
python baseCode/train_classification.py \
-vs 0.1 \
-ts 0.05 \
-lim 20000 \
-e 100 \
-bs 16 \
-w imagenet \
-lvl heavy \
-lr 0.001 \
-res checkpoints/convnext_small_clas_20251023_155914_MSI/best_model.pth \
-tb 

# TensorBoard (los logs están dentro del directorio del modelo)
tensorboard --logdir checkpoints/nombre_del_modelo/
# o para ver todos los modelos:
tensorboard --logdir checkpoints/

## Construcción de datasets desde CSV

El sistema incluye una utilidad modular para construir datasets desde archivos CSV:

```bash
# Ejemplo independiente
python baseCode/example_csv_builder.py

# Uso directo en entrenamiento
python baseCode/train_classification.py \
  --csv_metadata datos.csv \
  --images_dir imagenes/ \
  --allowed_labels Benign,Malignant \
  --val_split 0.1 \
  --test_split 0.05 \
  --limit 1000 \
  -b resnet50 -e 10 -tb
```

Ver `baseCode/utils/README_csv_builder.md` para documentación completa.