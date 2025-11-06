import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from models.classification import load_model
from argparse import ArgumentParser as argparse

# Nota: Este script debía cargar pesos entrenados, pero sólo construía la arquitectura.
# Esto produce un modelo ONNX con pesos aleatorios y explica la caída brutal de accuracy en OpenVINO.
# Se corrige cargando el checkpoint y tomando "effective_classes" si se entrenó en modo binary_sigmoid.

parser = argparse()
parser.add_argument('-m', '--model_folder', required=True, 
                    help='Path to model folder')
args = parser.parse_args()

# Read model parameters
weights = os.path.join(args.model_folder, 'best_model.pth')
config = os.path.join(args.model_folder, 'log.json')
cfg_dict = json.load(open(config))
backbone = cfg_dict["backbone"]
model_type = cfg_dict.get("model_type", "classification")
image_size = cfg_dict.get("image_size", 224)
binary_sigmoid = bool(cfg_dict.get("binary_sigmoid", False))
effective_classes = int(cfg_dict.get("effective_classes", cfg_dict.get("classes", 2)))
classes_original = int(cfg_dict.get("classes", effective_classes))

# Si hay modo binario + sigmoid y se entrenó con effective_classes=1, usamos esa dimensión de salida.
NUM_CLASSES = effective_classes
W = H = image_size
RESOLUTION = (W, H)
NUM_CHANNELS = 3

# Use cpu
device = 'cpu'

model = load_model(backbone, weights, NUM_CLASSES)
model.to(device)

# Cargar pesos reales del checkpoint (formato nuevo o antiguo)
checkpoint = torch.load(weights, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint cargado (formato completo).")
else:
    # Antiguo: directamente state_dict
    try:
        model.load_state_dict(checkpoint)
        print("State_dict cargado (formato simple).")
    except Exception as e:
        raise RuntimeError(f"No se pudieron cargar los pesos del modelo: {e}")

model.eval()

# Count number of parameters
n_params = sum(p.numel() for p in model.parameters())
print(f'{backbone}_{model_type} successfully loaded in {device}')
print('Number of parameters: {:d}'.format(n_params))
print(f'Original classes (log.json): {classes_original} | Effective classes used: {NUM_CLASSES}')
if binary_sigmoid:
    print('Modo binary_sigmoid activo -> salida de dimensión {}.'.format(NUM_CLASSES))
print('Image size: {}x{}'.format(W,H))

# Generate Dummy input
tform = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
dummy_input = (255*np.random.rand(H, W, 3)).astype('uint8')
dummy_input = Image.fromarray(dummy_input).resize(RESOLUTION)
dummy_input = tform(dummy_input).unsqueeze(0)  # [1,3,H,W]
print('Dummy input shape:', tuple(dummy_input.shape))

# Save with onnx format
name = os.path.basename(weights).split('.')[0] + '.onnx'
opt_path = os.path.join(args.model_folder, name)
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        opt_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes=None  # Batch fijo = 1 (puedes cambiar si necesitas variable)
    )
print('Model ONNX saved at:')
print(opt_path)

# Next step
print('\nPara convertir a OpenVINO ejecuta:')
print(f'mo --framework=onnx --input_model={opt_path} --input_shape=[1,{NUM_CHANNELS},{H},{W}] --output_dir={args.model_folder}')
print('\nVerifica que la dimensión de salida en el modelo IR sea {}.'.format(NUM_CLASSES))