import os
import json
import numpy as np
from argparse import ArgumentParser as argparse

import torch
from torch.utils.data import DataLoader

from models.siamese import siamese_embeddings
from dataloaders.Image_Dataset import Image_Dataset


def make_templates(train_list, model_folder, N_templates=4, model=None):

    # Configuration
    weights = os.path.join(model_folder, 'best_model.pth')
    npz_path = os.path.join(model_folder, 'templates.npz')
    config = os.path.join(model_folder, 'log.json')
    cfg_dict = json.load(open(config))
    backbone = cfg_dict["backbone"]
    image_size = cfg_dict["image_size"]

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nWorking with: {}".format(device))

    if model is None:
        # Load model
        model = model_embeddings(backbone, weights)
        model.to(device)

        # Load weights
        model.load_state_dict(torch.load(weights, map_location=device))
        print("\n{} model loaded from: \n{}".format(backbone, weights))

    # Make Dataloader
    dataset = Image_Dataset(train_list, img_size=image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    # Get a batch of bona fide images
    images = []
    cnt = 0

    for image, label, index in dataloader:
        lbl = label.item()
        if lbl == 0:
            images.append(image)
            cnt += 1
        if cnt >= N_templates:
            break

    batch = torch.cat(images, dim=0)
    batch.shape

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(batch.to(device)).detach().clone()

    # Convert to numpy
    templates = preds.cpu().numpy()

    # Save as npz
    np.savez(npz_path, templates=templates)
    print('\nTemplates saved at: \n{}\n'.format(npz_path))

    return preds


if __name__ == '__main__':
    parser = argparse()
    parser.add_argument('-l', '--train_list', type=str, required=True,
                        help='Path to the train list.')
    parser.add_argument('-m', '--model_folder', type=str, required=True,
                        help="Path to model folder.")
    parser.add_argument('-n', '--N_templates', type=int, default=4,
                        help='Number of templates to extract.')
    args = parser.parse_args()

    preds = make_templates(args.train_list, args.model_folder, args.N_templates)
