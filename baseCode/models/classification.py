import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    MaxVit_T_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)
from argparse import ArgumentParser as argparse


def replace_head_linear(module: nn.Module, classes: int) -> nn.Module:
    """Reemplaza una capa Linear manteniendo in_features.

    Acepta módulos tipo nn.Linear directamente.
    """
    if not isinstance(module, nn.Linear):
        raise TypeError("replace_head_linear espera un nn.Linear")
    in_feats = module.in_features
    return nn.Linear(in_feats, classes)


def replace_attr(model: nn.Module, attr_path: str, classes: int) -> nn.Module:
    """Reemplaza un atributo anidado (p.ej. 'fc', 'classifier.5', 'heads.head') por una nueva Linear.

    attr_path: cadena con puntos o índices de secuencial.
    """
    parts = attr_path.split('.')
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        target = parent[idx]
        parent[idx] = replace_head_linear(target, classes)
    else:
        target = getattr(parent, last)
        setattr(parent, last, replace_head_linear(target, classes))
    return model


def freeze_prefix(model: nn.Module, prefix_list):
    """Congela parámetros cuyo nombre contiene alguno de los prefijos dados."""
    for name, param in model.named_parameters():
        if any(pref in name for pref in prefix_list):
            param.requires_grad = False
    return model

def print_layers(model):
    children_counter = 0
    for n, c in model.named_children():
        print("Children Counter: ",children_counter, " Layer Name: ", n,)
        children_counter+=1
    return

MODEL_SPECS = {
    # CNN
    "alexnet": {"factory": models.alexnet, "head": "classifier.6", "weight_enum": "IMAGENET1K_V1"},
    "densenet121": {"factory": models.densenet121, "head": "classifier", "weight_enum": "IMAGENET1K_V1"},
    "densenet161": {"factory": models.densenet161, "head": "classifier", "weight_enum": "IMAGENET1K_V1"},
    "densenet169": {"factory": models.densenet169, "head": "classifier", "weight_enum": "IMAGENET1K_V1"},
    "densenet201": {"factory": models.densenet201, "head": "classifier", "weight_enum": "IMAGENET1K_V1"},
    "efficientnet_v2_l": {"factory": models.efficientnet_v2_l, "head": "classifier.1", "weight_enum": "DEFAULT"},
    "efficientnet_v2_m": {"factory": models.efficientnet_v2_m, "head": "classifier.1", "weight_enum": "DEFAULT"},
    "efficientnet_v2_s": {"factory": models.efficientnet_v2_s, "head": "classifier.1", "weight_enum": "DEFAULT"},
    "inception_v3": {"factory": models.inception_v3, "head": "fc", "weight_enum": "DEFAULT"},
    "maxvit_t": {"factory": models.maxvit_t, "head": "classifier", "weight_enum": MaxVit_T_Weights.IMAGENET1K_V1},
    "mobilenet_v2": {"factory": models.mobilenet_v2, "head": "classifier.1", "weight_enum": "IMAGENET1K_V2"},
    "mobilenet_v3_small": {"factory": models.mobilenet_v3_small, "head": "classifier.3", "weight_enum": "IMAGENET1K_V1"},
    "mobilenet_v3_large": {"factory": models.mobilenet_v3_large, "head": "classifier.3", "weight_enum": "IMAGENET1K_V2"},
    "resnet18": {"factory": models.resnet18, "head": "fc", "weight_enum": "IMAGENET1K_V1"},
    "resnet34": {"factory": models.resnet34, "head": "fc", "weight_enum": "IMAGENET1K_V1"},
    "resnet50": {"factory": models.resnet50, "head": "fc", "weight_enum": "IMAGENET1K_V2"},
    "resnet101": {"factory": models.resnet101, "head": "fc", "weight_enum": "IMAGENET1K_V2"},
    "resnet152": {"factory": models.resnet152, "head": "fc", "weight_enum": "IMAGENET1K_V2"},
    "vgg16": {"factory": models.vgg16, "head": "classifier.6", "weight_enum": "IMAGENET1K_V1"},
    "vgg19": {"factory": models.vgg19, "head": "classifier.6", "weight_enum": "IMAGENET1K_V1"},
    # Transformers / híbridos
    "vit_b_16": {"factory": models.vit_b_16, "head": "heads.head", "weight_enum": ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1},
    "vit_b_32": {"factory": models.vit_b_32, "head": "heads.head", "weight_enum": ViT_B_32_Weights.IMAGENET1K_V1},
    "swin_v2_t": {"factory": models.swin_v2_t, "head": "head", "weight_enum": Swin_V2_T_Weights.IMAGENET1K_V1},
    "swin_v2_s": {"factory": models.swin_v2_s, "head": "head", "weight_enum": Swin_V2_S_Weights.IMAGENET1K_V1},
    "swin_v2_b": {"factory": models.swin_v2_b, "head": "head", "weight_enum": Swin_V2_B_Weights.IMAGENET1K_V1},
    "convnext_tiny": {"factory": models.convnext_tiny, "head": "classifier", "weight_enum": ConvNeXt_Tiny_Weights.IMAGENET1K_V1},
    "convnext_small": {"factory": models.convnext_small, "head": "classifier", "weight_enum": ConvNeXt_Small_Weights.IMAGENET1K_V1},
    "convnext_base": {"factory": models.convnext_base, "head": "classifier", "weight_enum": ConvNeXt_Base_Weights.IMAGENET1K_V1},
    "convnext_large": {"factory": models.convnext_large, "head": "classifier", "weight_enum": ConvNeXt_Large_Weights.IMAGENET1K_V1},
}

TRANSFORMER_PREFIXES = ["vit", "swin_v2", "convnext", "maxvit_t"]

def resolve_weights(backbone: str, weights_str: str):
    if weights_str.lower() != "imagenet":
        return None
    spec = MODEL_SPECS.get(backbone.lower().replace('-', '_'))
    if not spec:
        return None
    return spec.get("weight_enum")


def partial_freeze_transformer(model, backbone):
    """Aplica congelamiento parcial en modelos tipo transformer.
    ConvNeXt: congela stages 0 y 1
    ViT: congela primeros 4 encoder layers (encoder.layers.0-3)
    Swin / MaxViT: congela primeros bloques según nombre.
    """
    b = backbone.lower().replace('-', '_')
    if b.startswith("convnext"):
        for name, p in model.features.named_parameters():
            if "stages.0" in name or "stages.1" in name:
                p.requires_grad = False
    elif b.startswith("vit_b"):
        # ViT estructura: encoder.layers
        for name, p in model.named_parameters():
            if "encoder.layers." in name:
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
                if layer_id < 4:
                    p.requires_grad = False
    elif b.startswith("swin_v2"):
        for name, p in model.named_parameters():
            # Congelar primeros dos stages aproximados
            if any(tag in name for tag in ["layers.0", "layers.1"]):
                p.requires_grad = False
    elif b.startswith("maxvit_t"):
        for name, p in model.named_parameters():
            if any(tag in name for tag in ["stages.0", "stages.1"]):
                p.requires_grad = False
    return model


def load_model(backbone, weights="None", classes=3, freeze_strategy="partial"):
    original_backbone = backbone
    backbone_norm = backbone.lower().replace('-', '_')
    spec = MODEL_SPECS.get(backbone_norm)
    if not spec:
        raise ValueError(f"Backbone no soportado: {original_backbone}")

    w_enum = resolve_weights(backbone_norm, weights)
    factory = spec["factory"]
    # Instancia con o sin pesos
    if w_enum is None:
        model = factory()
    else:
        model = factory(weights=w_enum)

    # Reemplazar head (Linear simple) para la mayoría de CNN / Swin / ViT cuando la ruta apunta a Linear
    head_path = spec["head"]
    # MaxViT / ConvNeXt necesitan tratamiento especial
    if backbone_norm.startswith("maxvit_t"):
        linear_indices = [i for i, m in enumerate(model.classifier) if isinstance(m, nn.Linear)]
        if not linear_indices:
            raise RuntimeError("No Linear en classifier de MaxViT")
        last_idx = linear_indices[-1]
        in_feats = model.classifier[last_idx].in_features
        model.classifier[last_idx] = nn.Linear(in_feats, classes)
    elif backbone_norm.startswith("convnext"):
        # Fine-tuning parcial
        if freeze_strategy == "partial":
            model = partial_freeze_transformer(model, backbone_norm)
        in_feats = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_feats, eps=1e-6),
            nn.Linear(in_feats, classes)
        )
    else:
        # Head genérico
        # Obtener módulo final
        parts = head_path.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        last = parts[-1]
        target = getattr(parent, last) if not last.isdigit() else parent[int(last)]
        if isinstance(target, nn.Linear):
            in_feats = target.in_features
            new_linear = nn.Linear(in_feats, classes)
            if last.isdigit():
                parent[int(last)] = new_linear
            else:
                setattr(parent, last, new_linear)
        else:
            # ViT heads.head puede ser Linear interno
            if backbone_norm.startswith("vit"):
                # fallback
                if hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
                    in_feats = model.heads.head.in_features
                    model.heads.head = nn.Linear(in_feats, classes)
                elif isinstance(model.heads[0], nn.Linear):
                    in_feats = model.heads[0].in_features
                    model.heads[0] = nn.Linear(in_feats, classes)
            else:
                raise RuntimeError(f"No se pudo adaptar head para {backbone_norm}")

    # Activar gradientes del nuevo head
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in ["classifier", "fc", "head", "heads.head"]):
            p.requires_grad = True

    # Congelamiento parcial para otros transformers (ViT, Swin, MaxViT) si aplica
    if freeze_strategy == "partial" and any(backbone_norm.startswith(pref) for pref in TRANSFORMER_PREFIXES):
        if not backbone_norm.startswith("convnext"):  # ya tratado
            model = partial_freeze_transformer(model, backbone_norm)

    return model


if __name__ == '__main__':
    parser = argparse()
    parser.add_argument('-b', '--backbone', type=str, default="vgg16",
                        help='Conv-Net backbone.')
    parser.add_argument('-w', '--weights', type=str, default="none",
                        help="Model's initial Weights: < none | imagenet | /path/to/weights/ >")
    parser.add_argument('-c', '--classes', type=int, default=5,
                        help='Number of output classes.')
    args = parser.parse_args()

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nWorking with: {}".format(device))

    # Get pretrained model
    model = load_model(args.backbone, args.weights, args.classes)
    model.to(device)
    print('{} model loaded on {} with weights: {}'.format(args.backbone, device, args.weights))

    print('\nLayers:')
    print_layers(model)