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

CLASSES_21 = [
'animal','beach','buildings','clouds','flowers','grass','lake','mountain','ocean','person',
'plants','reflection','road','rocks','sky','snow','sunset','tree','vehicle','water','window'
]
CLASSES_10= ['sky','clouds','person',  'water',  'animal','grass','buildings', 'window' ,'plants' , 'lake ']


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
all_label_features=generate_label(CLASSES_21)
print(1)


import pickle

P = os.getcwd()
print(P)
with open(join(P, "data/nus_21/label_features_21.pkl"), 'wb') as f:
    pickle.dump(all_label_features, f)

