from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from openvino.runtime import Core
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Ruta a los archivos del modelo
MODEL_XML = "checkpoints/convnext_tiny_clas_20251023_221506_MSI/best_model.xml"
MODEL_BIN = "checkpoints/convnext_tiny_clas_20251023_221506_MSI/best_model.bin"

# Inicializar OpenVINO
core = Core()
model = core.read_model(model=MODEL_XML, weights=MODEL_BIN)
compiled_model = core.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Preprocesamiento: resize y normalización (ajustar según modelo)
    image = image.resize((112, 112))
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # (C, H, W)
    img_np = np.expand_dims(img_np, axis=0)   # (1, C, H, W)
    # Inferencia
    result = compiled_model([img_np])[output_layer]
    # Procesar resultado (ajustar según modelo)
    pred_class = int(np.argmax(result))
    pred_score = float(np.max(result))
    class_names = {0: "Benign", 1: "Malignant"}
    return JSONResponse({
        "class": pred_class,
        "class_name": class_names.get(pred_class, "Unknown"),
        "score": pred_score
    })

# Bloque main para inicializar la API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_infer_openvino:app", host="0.0.0.0", port=8000, reload=True)
