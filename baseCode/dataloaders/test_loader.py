import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils import read_list
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

class TestLoader(Dataset):
    def __init__(self, list_path, img_size):
        # Get inputs
        self.img_size = img_size

        # Make transformation to convert to tensor
        self.to_tensor = v2.Compose([
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        # Read dataset from list
        img_paths, labels = read_list(list_path)
        self.images = img_paths

        # Get labels and indexes
        self.labels = np.array(labels, dtype=int)
        self.index = np.array([*range(len(self.labels))], dtype=int)

        # Set dataset length
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        # Get data
        label = self.labels[item]
        image = self.images[item]

        # Read Image
        image = Image.open(image).convert('RGB').resize(self.img_size)

        # Transform to tensor
        image = self.to_tensor(image)

        return image, label, item
