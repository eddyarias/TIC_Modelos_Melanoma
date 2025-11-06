import os
import json
import numpy as np
from openvino.runtime import Core
from tqdm import tqdm

from dataloaders.Image_Dataset import Image_Dataset
from torch.utils.data import DataLoader

def test_model_openvino(test_list, batch_size, num_workers, model_xml_path, model_bin_path, log_json_path, img_size=None):
    """Evalúa un modelo OpenVINO asegurando el mismo preprocesamiento que en PyTorch.
    Parámetros:
        test_list: ruta al archivo de lista de test.
        batch_size: tamaño de batch.
        num_workers: workers del DataLoader.
        model_xml_path / model_bin_path: archivos del modelo OpenVINO.
        log_json_path: ruta explícita al log.json del entrenamiento original.
        img_size: opcional; si None se obtiene de log.json.
    """
    # Leer config completa
    cfg = None
    if img_size is None:
        if not os.path.isfile(log_json_path):
            raise FileNotFoundError(f"No se encontró log.json en la ruta proporcionada: {log_json_path}")
        try:
            cfg = json.load(open(log_json_path, 'r'))
            img_size = int(cfg.get('image_size', 224))
            print(f"Usando image_size {img_size} leído de {log_json_path}")
        except Exception as e:
            raise RuntimeError(f"Error leyendo image_size de log.json: {e}")
    else:
        img_size = int(img_size)
        if os.path.isfile(log_json_path):
            try:
                cfg = json.load(open(log_json_path,'r'))
            except:
                cfg = None

    # Extraer flags de configuración para consistencia con PyTorch
    binary_sigmoid = False
    effective_classes = None
    classes = None
    if cfg:
        binary_sigmoid = bool(cfg.get('binary_sigmoid', False))
        effective_classes = int(cfg.get('effective_classes', cfg.get('classes', 1)))
        classes = int(cfg.get('classes', effective_classes or 1))
    # Mensajes informativos
    if effective_classes:
        print(f"Clases originales: {classes} | Clases efectivas: {effective_classes} | binary_sigmoid={binary_sigmoid}")

    # Dataset: Image_Dataset ya aplica PILToTensor + escala + Normalize(mean/std)
    test_dataset = Image_Dataset(test_list, img_size=(img_size, img_size), transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Cargar modelo OpenVINO
    ie = Core()
    model = ie.read_model(model=model_xml_path, weights=model_bin_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    y_true = []
    y_pred = []
    scores = []

    # Importante: las imágenes ya están normalizadas; solo convertir a numpy float32
    for images, labels, _ in tqdm(test_loader, desc="Test loop (OpenVINO)"):
        images_np = images.numpy().astype(np.float32)  # [B,3,H,W]
        result = compiled_model({input_layer: images_np})[output_layer]
        scores.append(result)
        if binary_sigmoid and (effective_classes == 1 or result.shape[1] == 1):
            # result shape [B,1] logits -> aplicar sigmoid para predicción binaria
            probs = 1/(1+np.exp(-result))
            preds = (probs >= 0.5).astype(int).squeeze(1)
        else:
            preds = np.argmax(result, axis=1)
        y_true.append(labels.numpy())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    scores = np.concatenate(scores)
    return y_true, y_pred, scores, {
        'binary_sigmoid': binary_sigmoid,
        'effective_classes': effective_classes,
        'classes': classes,
        'image_size': img_size
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tl', '--test_list', required=True, type=str, help='Ruta a la lista de test.')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('-j', '--jobs', default=1, type=int, help='Num workers.')
    parser.add_argument('-xml', '--model_xml', required=True, type=str, help='Ruta al modelo .xml de OpenVINO.')
    parser.add_argument('-bin', '--model_bin', required=True, type=str, help='Ruta al modelo .bin de OpenVINO.')
    parser.add_argument('-log', '--log_json', required=True, type=str, help='Ruta al archivo log.json del entrenamiento.')
    parser.add_argument('-sz', '--img_size', type=int, default=None, help='Override manual del tamaño de imagen (opcional).')
    args = parser.parse_args()

    y_true, y_pred, scores, meta = test_model_openvino(
        test_list=args.test_list,
        batch_size=args.batch_size,
        num_workers=args.jobs,
        model_xml_path=args.model_xml,
        model_bin_path=args.model_bin,
        log_json_path=args.log_json,
        img_size=args.img_size
    )
    acc = np.sum(y_true == y_pred) / len(y_pred)
    print(f"Test set accuracy: {acc:.4f}")
    # Guardar scores en la misma carpeta del log.json
    save_dir = os.path.dirname(args.log_json)
    save_path = os.path.join(save_dir, 'scores_openvino.npz')
    np.savez(save_path, y_true=y_true, y_pred=y_pred, scores=scores, **meta)
    print(f"Scores guardados en {save_path}")
