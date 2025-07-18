
import torch

def get_embedding_size(model, img_size, device):
    # Make blank image
    blank = torch.zeros((1, 3, img_size[0], img_size[1])).to(device)
    # Get model prediction
    model.eval()
    with torch.no_grad():
        features = model(blank)

    return features.shape[1]
