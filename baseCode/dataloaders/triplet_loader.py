import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils import read_list
from joblib import Parallel, delayed
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

class TipletLoader(Dataset):
    def __init__(self, list_path, args, keep_ratio=0.25, transform=None):
        # Get inputs
        self.img_size = (args.img_size,args.img_size)
        self.load_in_ram = args.load_in_ram
        self.repeat=args.multiply
        self.da_library = args.da_library
        self.jobs = args.jobs
        self.keep_ratio=keep_ratio
        self.transform = transform

        # Make transformation to convert to tensor
        self.to_tensor = v2.Compose([
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

        # Read dataset from list
        img_paths, labels = read_list(list_path)
        self.img_paths = img_paths #[:8*1024]

        # Get labels and indexes
        self.labels = np.array(labels, dtype=int)
        self.index = np.array([*range(len(self.labels))], dtype=int)

        # Load images
        self.load_images()

        # Set dataset length
        self.len = self.repeat*len(self.images)

        # Initialize Triplets
        self.triplets = np.zeros((self.len, 3), dtype=np.int32)
        self.hardest = np.zeros(self.len, dtype=np.bool_)
        self.update_triplets()

    def __len__(self):
        return self.len

    def apply_transform(self, image):
        if self.transform:
            if self.da_library in ['torchvision', 'pytorch', 'torch']:
                # Transform image using torchvision
                image = self.transform(image)
            else:
                # Convert to numpy and transform
                image = self.transform(image=np.array(image))
                # Obtain transformed image
                if self.da_library in ['albumentations', 'Albumentations']:
                    image = image['image']
                # Convert to pillow
                image = Image.fromarray(image)
        return image

    def load_image(self, i):
        pass
        # Read image
        img = Image.open(self.img_paths[i]).convert('RGB').resize(self.img_size)
        # Apply transformation
        img = self.apply_transform(img)
        # Transform to tensor
        img = self.to_tensor(img).unsqueeze(0)

        return img

    def load_images(self):
        self.images = None
        if self.load_in_ram:
            self.images = Parallel(n_jobs=self.jobs)(
                                delayed(self.load_image)(i)
                                for i in tqdm(self.index, desc='Loading imgs in RAM')
                                )
            self.images = torch.cat(self.images, dim=0)
        else:
            self.images = self.img_paths
        return

    def make_triplet(self, item):
        anchor_label = self.labels[item]

        positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]
        positive_item = random.choice(positive_list)

        negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
        negative_item = random.choice(negative_list)

        return item, positive_item, negative_item

    def update_triplets(self, losses=None):
        if losses is not None:
            limit = np.sort(losses)[-int(self.keep_ratio*len(losses))]
            self.hardest = losses >= limit

        i = 0
        #for item in tqdm(range(len(self.images)), desc="Updating  triplets "):
        for item in range(len(self.images)):
            for x in range(self.repeat):
                if not self.hardest[i]:
                    self.triplets[i, :] = self.make_triplet(item)
                i += 1
        return

    def __getitem__(self, item):
        # Get triplet
        anchor_item, positive_item, negative_item = self.triplets[item]

        # Get label
        anchor_label = self.labels[anchor_item]

        # Get images
        anchor_img = self.images[anchor_item]
        positive_img = self.images[positive_item]
        negative_img = self.images[negative_item]

        if not self.load_in_ram:
            # Read Image
            anchor_img = Image.open(anchor_img).convert('RGB').resize(self.img_size)
            positive_img = Image.open(positive_img).convert('RGB').resize(self.img_size)
            negative_img = Image.open(negative_img).convert('RGB').resize(self.img_size)

            # Apply Transformation
            anchor_img = self.apply_transform(anchor_img)
            positive_img = self.apply_transform(positive_img)
            negative_img = self.apply_transform(negative_img)

            # Transform to tensor
            anchor_img = self.to_tensor(anchor_img)
            positive_img = self.to_tensor(positive_img)
            negative_img = self.to_tensor(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label, item
