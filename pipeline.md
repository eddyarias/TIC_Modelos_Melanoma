# Pipeline de Clasificación de Lesiones Cutáneas

Este documento describe con detalle el pipeline completo empleado para el prototipo de clasificación de lesiones dermatológicas (Benigno/Maligno) siguiendo la metodología de prototipado de Pressman. Incluye la planificación rápida, requisitos del modelo, arquitectura propuesta (basada en Transformers/CNN modernas), flujo de datos, entrenamiento, validación, pruebas, y despliegue. Se agregan además descripciones de gráficos sugeridos para documentación.

---

## 2.2 Fase 2 – Planificación rápida

Objetivo: definir un plan preliminar que guíe la construcción del prototipo, de bajo riesgo y rápidamente iterativo.

- **Necesidades de datos:**
  - Listas explícitas `lists/train.txt`, `lists/validation.txt`, `lists/test.txt` con pares `ruta_imagen etiqueta`.
  - Mapeo `lists/label_mapping.json` con correspondencia de clase → índice.
  - Estadísticas `lists/stats.json` (distribución de clases, conteos por split) para monitoreo.
- **Alcance técnico mínimo viable:**
  - Entrenar un modelo de visión con transferencia de aprendizaje (ConvNeXt, ViT, Swin V2) usando PyTorch.
  - Preprocesamiento estandarizado ImageNet (normalización y tamaño fijo).
  - Métricas de entrenamiento y validación registradas en `TensorBoard` y `checkpoints/.../log.json`.
  - Pruebas de inferencia en CPU/GPU (PyTorch) y OpenVINO.
- **Iteraciones rápidas:**
  - Selección de backbone, hiperparámetros básicos (épocas, batch size, LR), y validación temprana.
  - Guardado de `best_model.pth` y `last_model.pth` para comparación.

### Entregables de la fase
- Directorio `checkpoints/<modelo_fecha_host>/` con:
  - `best_model.pth`, `last_model.pth`, `log.json`, `tensorboard_logs/`, `lists/` (si aplica).
- Reporte inicial de precisión y curvas de pérdida/accuracy.

---

## 2.2.1 Requisitos del modelo

- **Entrada:** una imagen RGB (JPG/PNG) de una lesión cutánea, reescalada a `img_size × img_size` (por defecto 224).
- **Procesamiento:**
  - Normalización con medias/varianzas de ImageNet.
  - Paso por el backbone seleccionado (ConvNeXt/ViT/Swin/ResNet/VGG) con cabeza de clasificación adaptada al número de clases.
  - Opción binaria con `--binary_sigmoid`: para problemas de 2 clases, la cabeza produce un único logit y se usa `BCEWithLogitsLoss`.
- **Salida:**
  - Clase predicha: índice → etiqueta vía `label_mapping.json`.
  - Puntaje de confianza:
    - Multiclase: `softmax(logits)`.
    - Binario: `sigmoid(logit)` interpretado como probabilidad de Maligno (o según convención definida).
- **Criterios binarios:**
  - Umbral por defecto 0.5 para decisión; configurable si se requiere sensibilidad/especificidad distinta.

---

## 2.2.2 Arquitectura del modelo Transformer propuesto

Aunque el pipeline soporta múltiples backbones, se propone una arquitectura basada en **Transformers de visión** (ViT/Swin/ConvNeXt) por su rendimiento en clasificación de imágenes.

- **Limitaciones de enfoques tradicionales (CNN puras):**
  - Receptive fields locales: pueden perder relaciones globales en texturas y patrones sutiles.
  - Escalado de profundidad/anchura incrementa costo computacional sin mejorar la captura de dependencias de largo alcance.
- **Ventajas de Transformers/ConvNeXt:**
  - Atención (auto-atención o variantes jerárquicas) que modelan relaciones de largo alcance.
  - Diseño moderno con mejoras en normalización, bloques y entrenamiento estable.
  - Resultados competitivos en benchmarks, buena transferibilidad y compatibilidad con preentrenamiento.

### Componentes arquitectónicos
- **Backbone:** seleccionable vía argumento `--backbone`:
  - `convnext_tiny/small/base/large`: CNN moderna inspirada en diseños tipo transformer.
  - `vit_b_16/b_32`: Vision Transformer con cabeza `heads.head` reemplazada.
  - `swin_v2_t/s/b`: Transformer jerárquico con ventanas deslizantes.
  - También disponibles CNN clásicas: `resnet50`, `vgg16`, etc.
- **Cabeza de clasificación:** reemplazo de la última capa `Linear` según el número de clases detectado o indicado.
- **Congelamiento parcial (opcional):** primeras etapas/layers del backbone pueden congelarse para estabilizar el fine-tuning rápido.
- **Pérdida:**
  - Multiclase: `CrossEntropyLoss`.
  - Binaria (con `--binary_sigmoid`): `BCEWithLogitsLoss` con salida de 1 logit.
- **Optimización:** `AdamW`/`SGD` (según configuración), con scheduler opcional.

### Flujo de datos y entrenamiento
- **Datos (`Image_Dataset`):** lee listas, carga imágenes con PIL, aplica resize, normaliza con `torchvision.transforms.v2`.
- **Aumento de datos:** `dataloaders/data_augmentation.py` selecciona librerías (`torchvision`/`albumentations`) y niveles (`light/medium/heavy`).
- **Entrenamiento (`train_classification.py`):**
  1. Configura rutas de salida en `checkpoints/<model_name>/`.
  2. Carga listas (`lists/*.txt`) y mapeo (`label_mapping.json`).
  3. Construye `DataLoader`s para train/val.
  4. Inicializa modelo (`models/classification.load_model`).
  5. Ejecuta bucles de `train` y `validate`, guarda el mejor modelo y logs.
  6. Opción `--tensorboard` para visualización de métricas.
- **Validación y prueba:**
  - `test_classification.py`: evalúa el modelo PyTorch, calcula accuracy y guarda `scores.npz`.
  - `test_classification_openvino.py`: evalúa exportaciones OpenVINO con mismo preprocesamiento.

### Complejidad y componentes tecnológicos necesarios
- **Tecnologías:** PyTorch, TorchVision, Albumentations (opcional), NumPy, TQDM, TensorBoard, OpenVINO (para evaluación), Matplotlib.
- **Recursos de cómputo:** GPU NVIDIA (CUDA) preferible; CPU posible con tiempos mayores.
- **Estructura de proyecto:**
  - `baseCode/models/classification.py`, `baseCode/train_classification.py`, `baseCode/dataloaders/Image_Dataset.py`, `baseCode/utils/*`, `lists/*`.

### Funciones mínimas requeridas del prototipo
- Entrenar con listas predefinidas y mapeo de etiquetas.
- Reportar métricas de entrenamiento/validación (pérdida, accuracy).
- Inferir sobre conjunto de prueba y exportar puntajes.
- Soporte binario con `--binary_sigmoid` y ajuste de cabeza.
- Registro de configuración y checkpoints.

### Bases de datos públicas de imágenes dermatológicas (identificación)
- **ISIC Archive**: colecciones con etiquetas benigno/maligno y metadatos.
- **Derm7pt**: dataset con siete criterios de diagnóstico.
- **PH^2**: conjunto con lesiones melanocíticas con segmentaciones.
- **HAM10000**: 10k imágenes de lesiones con múltiples clases; útil para transfer learning.

---

## Gráficos y diagramas sugeridos

Para documentar el pipeline y la arquitectura, se proponen los siguientes gráficos. Puedes generarlos con herramientas como draw.io, Mermaid, o PowerPoint.

1. **Diagrama de flujo del pipeline (alto nivel)**
   - Cajas: `lists/*.txt` → `Image_Dataset` → `DataLoader` → `Backbone + Head` → `Loss/Optimizer` → `Checkpoints/Logs` → `Test/Scores`.
   - Flechas con notas: normalización, DA al entrenar, validación por época, selección de `best_model`.

2. **Arquitectura del modelo (Transformer/CNN)**
   - Bloques: Patch Embedding / Stages (según backbone) → Bloques (MHSA/ConvNext Block) → Global Pool → Linear Head.
   - Nota de configuración: reemplazo de la `Linear` final acorde a `classes` o modo binario.

3. **Preprocesamiento y aumento de datos**
   - Ilustrar resize a `img_size`, normalización ImageNet.
   - Árbol de decisión de DA: `torchvision` vs `albumentations`, niveles `light/medium/heavy`.

4. **Curvas de entrenamiento**
   - Gráfica de `Train/Val Loss` y `Train/Val Accuracy` por época (sugerencia: `TensorBoard` o `Matplotlib`).

5. **Distribución de clases**
   - Barras de conteos por clase (a partir de `stats.json`).
   - Pie (opcional) para proporción por split.

6. **Flujo de inferencia (PyTorch y OpenVINO)**
   - PyTorch: `test_list` → `Image_Dataset` → `Model` → `softmax/sigmoid` → `scores.npz`.
   - OpenVINO: `XML+BIN` → `Runtime` → `CompiledModel` → salida y métricas.

---

## Ejecución del pipeline (resumen)

- **Entrenamiento (desde raíz del proyecto):**
  ```cmd
  python baseCode\train_classification.py --dataset lists --backbone convnext_tiny --epochs 10 --batch_size 32 --jobs 8 --tensorboard --binary_sigmoid
  ```
- **Prueba PyTorch:**
  ```cmd
  python baseCode\test_classification.py -m checkpoints\<tu_modelo> -l lists\test.txt -bs 32 -j 8
  ```
- **Prueba OpenVINO:**
  ```cmd
  python baseCode\test_classification_openvino.py -xml <ruta_modelo.xml> -bin <ruta_modelo.bin> -log checkpoints\<tu_modelo>\log.json -bs 32 -j 8
  ```

---

## Consideraciones finales

- Mantener consistencia de preprocesamiento entre entrenamiento y pruebas.
- Verificar el `label_mapping.json` para interpretar correctamente las clases en informes.
- Ajustar el umbral de decisión en el modo binario según métricas clínicas (sensibilidad/especificidad).
- Usar `TensorBoard` para monitorear y decidir early stopping si se requiere.
