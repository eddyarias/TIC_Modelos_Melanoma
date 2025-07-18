import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser as argparse

import torch
from torch.utils.data import DataLoader

from models.classification import load_model
from dataloaders.Image_Dataset import Image_Dataset
from utils.score_normalization  import analyze_scores, normalize_scores

def test_model(test_list, model_folder, batch_size, jobs, model=None, k=5):

    # Configuration
    weights = os.path.join(model_folder, 'best_model.pth')
    scores_path = os.path.join(model_folder, 'scores.npz')
    config = os.path.join(model_folder, 'log.json')
    cfg_dict = json.load(open(config))
    backbone = cfg_dict["backbone"]
    image_size = cfg_dict["image_size"]
    classes = cfg_dict["classes"]

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nWorking with: {}".format(device))

    if model is None:
        # Load model
        model = load_model(backbone, weights, classes)
        model.to(device)

        # Load weights
        model.load_state_dict(torch.load(weights, map_location=device))
        print("\n{} model loaded from: \n{}".format(backbone, weights))

    # Make Dataloader
    dataset = Image_Dataset(test_list, img_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=jobs)

    # Make predictions
    model.eval()
    scores = np.zeros((len(dataset),classes), dtype=float)
    labels = np.zeros(len(dataset), dtype=int)
    with torch.no_grad():
        for image, label, index in tqdm(dataloader):
            preds = model(image.to(device)).detach().cpu().clone()
            for i in range(label.shape[0]):
                idx = index[i].item()
                scores[idx] = preds[i].numpy()
                labels[idx] = label[i].item()

    # Normalize scores
    lim_l, lim_u = analyze_scores(scores, labels)
    norm_scores = normalize_scores(scores, lim_l, lim_u, k)

    # Save scores
    scores_path = os.path.join(model_folder, 'scores.npz')
    np.savez(scores_path, scores=scores, labels=labels, dataset=test_list, images=dataset.images,
             norm_scores=norm_scores, normalization=[lim_l, lim_u, k] )
    print('\nScores saved at: \n{}\n'.format(scores_path))
    
    # Save normalization in the config file
    cfg_dict["normalization"] = [lim_l, lim_u, k]
    with open(config, 'w') as write_file:
        json.dump(cfg_dict, write_file, indent=4)

    return labels, np.argmax(scores,axis=1)


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
