import json
import numpy as np

def read_txt(path):
    text = []
    with open(path) as f:
        for line in f:
            text.append(line.split('\n')[0])
    N = len(text)
    return text, N

def read_indices(path):
    indices = []
    with open(path) as f:
        for line in f:
            indices.append([int(x) for x in line.split(',')])
    N = len(indices)
    return indices, N

def read_num(path):
    numbers = []
    with open(path) as f:
        for line in f:
            numbers.append(float(line))
    N = len(numbers)
    return numbers, N

def read_list(list_path, limit=None):
    text, N = read_txt(list_path)
    if limit is not None:
        np.random.shuffle(text)
        text = text[:limit]
        N = len(text)
    img_paths = []
    labels = []
    for line in text:
        img, lbl = line.split(' ')
        img_paths.append(img)
        labels.append(int(lbl))
    return img_paths, labels

def write_txt(lst, opt_path):
    # open file in write mode
    with open(opt_path, 'w') as fp:
        for item in lst:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(f'Text file saved at: {opt_path}')
    return

def write_json(dictionary, opt_path):
    with open(opt_path, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)
    return
