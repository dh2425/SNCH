import json
import random
import numpy as np
import torch
from os.path import join
from PIL import Image
import clip
from numpy import dtype
from tqdm import tqdm
import os
import json

def seed_torch(seed=27):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

def generate_label(CLASSES):
    N_tag= len(CLASSES)
    all_label_features = np.zeros((N_tag, 512))
    with torch.no_grad():
        for i, data in enumerate(CLASSES):
            text = clip.tokenize(data).to(device)
            text_features = model.encode_text(text).cpu().numpy()  # （1,512）
            one_sample = np.array(text_features)
            all_label_features[i, :] = one_sample
    return all_label_features
all_label_features=generate_label(CLASSES)
print(1)


import pickle

P = os.getcwd()
print(P)
with open(join(P, "data/coco/label_features.pkl"), 'wb') as f:
    pickle.dump(all_label_features, f)

