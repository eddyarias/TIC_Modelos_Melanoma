from PIL import Image
import albumentations as A

interp = {
    "cv2_INTER_NEAREST" : 0,
    "cv2_INTER_LINEAR"  : 1,
    "cv2_INTER_CUBIC"   : 2,
    "cv2_INTER_AREA"    : 3,
    "cv2_INTER_LANCZOS4": 4
}

def build_augmentation_sequence(level):
    if level in ['light','Light','LIGHT',1,'1']:
        return data_augmentation_light()
    elif level in ['medium','Medium','MEDIUM',2,'2']:
        return data_augmentation_medium()
    elif level in ['heavy','Heavy','HEAVY',3,'3']:
        return data_augmentation_heavy()

def data_augmentation_light(
        affine_chance=0.20,
        brightness_chance=0.35,
        blur_chance=0.35,
        noise_chance=0.05
        ):

    transform = A.Compose([
        #A.Resize (448, 448, interpolation=interp["cv2_INTER_AREA"], p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.15),
                                   contrast_limit=(-0.2, 0.2),
                                   p=brightness_chance),
        A.MedianBlur(blur_limit=(3, 5),p=blur_chance)
    ])

    return transform

def data_augmentation_medium(
        affine_chance=0.20,
        brightness_chance=0.35,
        blur_chance=0.35,
        noise_chance=0.05
        ):

    transform = A.Compose([
        #A.Resize (448, 448, interpolation=interp["cv2_INTER_AREA"], p=1),
        A.Affine(rotate=(-12, 12), p=affine_chance),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.15),
                                   contrast_limit=(-0.2, 0.2),
                                   p=brightness_chance),
        A.MedianBlur(blur_limit=(3, 5),p=blur_chance),
        A.GaussNoise(p=noise_chance)
    ])

    return transform

def data_augmentation_heavy(
        affine_chance=0.20,
        brightness_chance=0.35,
        blur_chance=0.35,
        noise_chance=0.05
        ):

    transform = A.Compose([
        #A.Resize (448, 448, interpolation=interp["cv2_INTER_AREA"], p=1),
        A.Affine(rotate=(-20, 20), p=affine_chance),
        A.Affine(shear=(-12, 12), p=affine_chance),
        A.ColorJitter(brightness=(0.5, 1.1), contrast=0.25,
                      saturation=0.1, hue=0.1,
                      p=brightness_chance),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(1, 5),
                       p=blur_chance),
        A.MedianBlur(blur_limit=(3, 5),p=blur_chance),
        A.GaussNoise(p=noise_chance),
        A.RandomRain(p=noise_chance)
    ])

    return transform
