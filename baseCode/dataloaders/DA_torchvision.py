import torch
from PIL import Image
from torchvision.transforms import v2

def build_augmentation_sequence(level):
    if level in ['light','Light','LIGHT',1,'1']:
        return data_augmentation_light()
    elif level in ['medium','Medium','MEDIUM',2,'2']:
        return data_augmentation_medium()
    elif level in ['heavy','Heavy','HEAVY',3,'3']:
        return data_augmentation_heavy()

def data_augmentation_light(
            affine_chance=0.5,
            brightness_chance=0.5,
            temperature_chance=0.35,
            blur_chance=0.5):

    transform = v2.Compose([
        #v2.Resize(size=(448,448)),
        #v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply(
                [v2.ColorJitter(brightness=(0.6, 1.1), contrast=0.3)],
                p=brightness_chance),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform

def data_augmentation_medium(
            affine_chance=0.5,
            brightness_chance=0.5,
            temperature_chance=0.35,
            blur_chance=0.5):

    transform = v2.Compose([
        #v2.Resize(size=(448,448)),
        #v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([
            v2.RandomAffine(degrees=12, shear=8),
        ], p=affine_chance),
        v2.RandomApply([v2.RandomChoice([
            v2.ColorJitter(brightness=(0.5, 1.5)),
            v2.ColorJitter(contrast=0.3),
        ])], p=brightness_chance),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=(3,5), sigma=(1,3))],
            p=blur_chance),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform

def data_augmentation_heavy(
            affine_chance=0.5,
            brightness_chance=0.5,
            temperature_chance=0.35,
            blur_chance=0.5):

    transform = v2.Compose([
        #v2.Resize(size=(448,448)),
        #v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([v2.RandomChoice([
            v2.RandomAffine(degrees=25),
            v2.RandomAffine(degrees=0, shear=16),
            v2.RandomAffine(degrees=12, shear=8),
        ])], p=affine_chance),
        v2.RandomApply([v2.RandomChoice([
            v2.ColorJitter(brightness=(0.5, 1.5)),
            v2.ColorJitter(contrast=0.3),
        ])], p=brightness_chance),
        v2.RandomApply(
            [v2.ColorJitter(hue=0.1, saturation=0.1)],
            p=temperature_chance),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=(3,5), sigma=(1,3))],
            p=blur_chance),
        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform
