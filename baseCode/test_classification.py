import os
import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser as argparse

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
    binary_sigmoid = bool(cfg_dict.get("binary_sigmoid", False))
    effective_classes = int(cfg_dict.get("effective_classes", classes))

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nWorking with: {}".format(device))

    if model is None:
        # Load model (usar effective_classes para reconstruir arquitectura correcta)
        model = load_model(backbone, weights, effective_classes)
        model.to(device)

        # Load weights - handle both checkpoint format and old state_dict format
        print("\n{} model loaded from: \n{}".format(backbone, weights))
        try:
            checkpoint = torch.load(weights, map_location=device)
            if 'model_state_dict' in checkpoint:
                # New checkpoint format
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded from checkpoint format")
            else:
                # Old state_dict format
                model.load_state_dict(checkpoint)
                print("✅ Loaded from state_dict format")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            raise

    # Make Dataloader
    dataset = Image_Dataset(test_list, img_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=jobs)

    # Make predictions
    model.eval()
    if binary_sigmoid and effective_classes == 1:
        scores = np.zeros((len(dataset), 1), dtype=float)
    else:
        scores = np.zeros((len(dataset), classes), dtype=float)
    labels = np.zeros(len(dataset), dtype=int)
    with torch.no_grad():
        for image, label, index in tqdm(dataloader):
            logits = model(image.to(device)).detach().cpu().clone()
            if binary_sigmoid and effective_classes == 1:
                probs = torch.sigmoid(logits)
                for i in range(label.shape[0]):
                    idx = index[i].item()
                    scores[idx,0] = probs[i].item()
                    labels[idx] = label[i].item()
            else:
                for i in range(label.shape[0]):
                    idx = index[i].item()
                    scores[idx] = logits[i].numpy()
                    labels[idx] = label[i].item()

    # Normalize scores
    if binary_sigmoid and effective_classes == 1:
        # Para binario, podemos construir matriz 2-col para análisis de normalización si se requiere
        # Convertir prob a [p0, p1] = [1-p, p]
        probs = scores[:,0]
        two_col = np.vstack([1.0 - probs, probs]).T
        lim_l, lim_u = analyze_scores(two_col, labels)
        norm_scores = normalize_scores(two_col, lim_l, lim_u, k)
    else:
        lim_l, lim_u = analyze_scores(scores, labels)
        norm_scores = normalize_scores(scores, lim_l, lim_u, k)

    # Save scores
    scores_path = os.path.join(model_folder, 'scores.npz')
    np.savez(scores_path, scores=scores, labels=labels, dataset=test_list, images=dataset.images,
             norm_scores=norm_scores, normalization=[lim_l, lim_u, k],
             binary_sigmoid=binary_sigmoid, effective_classes=effective_classes )
    print('\nScores saved at: \n{}\n'.format(scores_path))
    
    # Save normalization in the config file
    cfg_dict["normalization"] = [lim_l, lim_u, k]
    with open(config, 'w') as write_file:
        json.dump(cfg_dict, write_file, indent=4)

    if binary_sigmoid and effective_classes == 1:
        preds = (scores[:,0] >= 0.5).astype(int)
        return labels, preds
    else:
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

    labels, preds = test_model(args.test_list, args.model_folder, args.batch_size, args.jobs)
    acc = np.sum(labels == preds) / len(preds)
    print(f"Test set accuracy: {acc:.4f}")
