import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser as argparse

import torch
from torch.utils.data import DataLoader

from models.siamese import siamese_embeddings
from dataloaders.Image_Dataset import Image_Dataset

def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1).sqrt().mean(0)

def test_model(test_list, model_folder, batch_size, jobs, model=None, templates=None):

    # Configuration
    weights = os.path.join(model_folder, 'best_model.pth')
    templates_path = os.path.join(model_folder, 'templates.npz')
    scores_path = os.path.join(model_folder, 'scores.npz')
    config = os.path.join(model_folder, 'log.json')
    cfg_dict = json.load(open(config))
    backbone = cfg_dict["backbone"]
    image_size = cfg_dict["image_size"]

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nWorking with: {}".format(device))

    if model is None:
        # Load model
        model = siamese_embeddings(backbone, weights)
        model.to(device)

        # Load weights
        model.load_state_dict(torch.load(weights, map_location=device))
        print("\n{} model loaded from: \n{}".format(backbone, weights))

    # Make Dataloader
    dataset = Image_Dataset(test_list, img_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=jobs)

    if templates is None:
        # Load templates
        templates = np.load(templates_path)["templates"]
        templates = torch.tensor(templates, device=device)
        print("\nTemplates loaded from: \n{}\n".format(templates_path))

    # Make predictions
    model.eval()
    dists = np.zeros(len(dataset), dtype=float)
    labels = np.zeros(len(dataset), dtype=int)
    with torch.no_grad():
        for image, label, index in tqdm(dataloader):
            preds = model(image.to(device)).detach().clone()
            for i in range(label.shape[0]):
                idx = index[i].item()
                dist = calc_euclidean(templates, preds[1].unsqueeze(0))
                dists[idx] = dist.item()
                labels[idx] = label[i].item()

    # Save scores
    scores_path = os.path.join(model_folder, 'scores.npz')
    np.savez(scores_path, scores=dists, labels=labels, dataset=test_list, images=dataset.images)
    print('\nScores saved at: \n{}\n'.format(scores_path))
    
    return dists, labels


if __name__ == '__main__':
    parser = argparse()
    parser.add_argument('-l', '--test_list', type=str, required=True,
                        help='Path to the test list.')
    parser.add_argument('-m', '--model_folder', type=str, required=True,
                        help="Path to model folder.")
    parser.add_argument('-bs', '--batch_size', type=int, default=24,
                        help='Batch size.')
    parser.add_argument('-j', '--jobs', type=int, default=6,
                        help="Number of workers for dataloader's parallel jobs.")
    args = parser.parse_args()

    dists, labels = test_model(args.test_list, args.model_folder, args.batch_size, args.jobs)
