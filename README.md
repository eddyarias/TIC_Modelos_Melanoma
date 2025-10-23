# TIC_Modelos_Melanoma
En este respositorio contiene los script de preprocesamiento, entrenamiento y evaluación de los modelos CNN y Transformer


directorio baseCode: contiene el codigo basico para entrenamientos de redes neuronales y transformers

directorio database: contine las etiquetas y los datos para entrenamiento, prueba y validación

# ejecutar un entrenamiento
python baseCode/train_classification.py \
  --csv_metadata ../DataTIC/bcn20000_metadata_2025-10-19.csv \
  --images_dir ../DataTIC/ISIC-images \
  --allowed_labels Benign,Malignant \
  --label_col diagnosis_1 \
  --image_id_col isic_id \
  --val_split 0.1 \
  --test_split 0.05 \
  --limit 20000 \
  -b convnext_large \
  -e 25 \
  -bs 16 \
  -w imagenet \
  -lvl heavy \
  -lr 0.001 \
  -lrf 0.5  \
  -tb

# comenzar desde un checkpoint
python baseCode/train_classification.py \
--csv_metadata ../DataTIC/bcn20000_metadata_2025-10-19.csv \
--images_dir ../DataTIC/ISIC-images \
--allowed_labels Benign,Malignant \
--label_col diagnosis_1 \
--image_id_col isic_id \
--val_split 0.1 \
--test_split 0.05 \
--limit 20000 \
-e 100 \
-bs 16 \
-w imagenet \
-lvl heavy \
-lr 0.001 \
--resume checkpoints/convnext_small_clas_20251023_155914_MSI/best_model.pth \
-tb 

# TensorBoard (los logs están dentro del directorio del modelo)
tensorboard --logdir checkpoints/nombre_del_modelo/
# o para ver todos los modelos:
tensorboard --logdir checkpoints/