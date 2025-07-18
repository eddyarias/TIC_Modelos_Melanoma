import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Normalization Function
def normalize_scores(scores, lim_l, lim_u, k=5):
    inrerval = lim_u - lim_l
    mean = (lim_u + lim_l)/2
    aux = k * (scores - mean) / inrerval
    return sigmoid(aux)

def analyze_scores(scores, gt_lbl):
    # Get scores of the first class vs the rest
    scores_A = scores[gt_lbl==0][:,0]
    scores_B = scores[gt_lbl!=0][:,0]
    # Get median of both distributions
    median_A = np.median(scores_A)
    median_B = np.median(scores_B)
    # Get lower and upper limits
    lim_u = max([median_A, median_B])
    lim_l = min([median_A, median_B])
    return lim_l, lim_u