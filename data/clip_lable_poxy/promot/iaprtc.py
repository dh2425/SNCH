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

CLASSES = ['adult', 'airplane', 'airport', 'anorak', 'area', 'back', 'backpack', 'bag', 'balcony', 'bank', 'bar', 'base', 'bay',
           'beach', 'bed', 'bedcover', 'bedside', 'bell', 'bench', 'bicycle', 'bike', 'bird', 'bit', 'blanket', 'bloom', 'board',
           'boat', 'body', 'bone', 'bottle', 'boy', 'branch', 'brick', 'bridge', 'building', 'bus', 'bush', 'cactus', 'camera',
           'canyon', 'cap', 'cape', 'car', 'carpet', 'cathedral', 'ceiling', 'centre', 'chair', 'child', 'church', 'city', 'classroom',
           'cliff', 'clock', 'cloth', 'clothes', 'cloud', 'coast', 'cobblestone', 'column', 'condor', 'corner', 'corridor', 'couch',
           'country', 'couple', 'court', 'courtyard', 'creek', 'cross', 'cup', 'curtain', 'cycling', 'cyclist', 'deck', 'desert', 'desk',
           'dirt', 'dog', 'dome', 'door', 'dress', 'dune', 'edge', 'embankment', 'entrance', 'face', 'fence', 'fern', 'field', 'fjord', 'flag',
           'flagpole', 'floor', 'flower', 'fog', 'footpath', 'forest', 'formation', 'fountain', 'frame', 'front', 'fruit', 'garden', 'gate',
           'giant', 'girl', 'glacier', 'glass', 'grandstand', 'grass', 'grave', 'gravel', 'green', 'grey', 'ground', 'group', 'hair', 'hall',
           'hammock', 'hand', 'harbour', 'hat', 'head', 'hedge', 'helmet', 'highway', 'hill', 'horizon', 'horse', 'house', 'hut', 'island',
           'jacket', 'jean', 'jeep', 'jersey', 'jetty', 'jumper', 'jungle', 'kid', 'lagoon', 'lake', 'lamp', 'landscape', 'lane', 'lawn',
           'leave', 'leg', 'level', 'life', 'light', 'line', 'lion', 'llama', 'lookout', 'lot', 'luggage', 'man', 'meadow', 'middle',
           'monument', 'mountain', 'mummy', 'neck', 'net', 'night', 'one', 'orange', 'ornament', 'painting', 'palm', 'pant', 'park',
           'path', 'pavement', 'paving', 'peak', 'penguin', 'people', 'person', 'photo', 'picture', 'pillow', 'pinnacle', 'plane', 'plant',
           'plate', 'player', 'pole', 'polo', 'pond', 'pool', 'port', 'portrait', 'pot', 'power', 'pullover', 'racetrack', 'racing', 'rack',
           'rail', 'railing', 'range', 'ravine', 'red', 'reed', 'reflection', 'restaurant', 'ridge', 'river', 'road', 'rock', 'roof', 'room',
           'rope', 'round', 'ruin', 'salt', 'sand', 'sea', 'seat', 'shade', 'shelf', 'shelter', 'ship', 'shirt', 'shoe', 'shop', 'shore', 'short',
           'shrub', 'side', 'sign', 'skirt', 'skull', 'sky', 'skyline', 'skyscraper', 'slope', 'snow', 'sock', 'space', 'spectator', 'square',
           'stadium', 'stage', 'stair', 'stand', 'statue', 'stone', 'street', 'stripe', 'summit', 'sun', 'sunset', 'surfer', 'sweater', 'table',
           'table-cloth', 'team', 'tee-shirt', 'tennis', 'tent', 'terrace', 'terrain', 'tile', 'tourist', 'towel', 'tower', 'trail', 'train', 'tree',
           'trouser', 'trunk', 'tussock', 'uniform', 'valley', 'vegetation', 'view', 'village', 'waistcoat', 'wall', 'water', 'waterfall', 'wave',
           'white', 'window', 'woman', 'wood', 'writing']

def generate_label(CLASSES):
    N_tag= len(CLASSES)
    all_label_features = np.zeros((N_tag, 512))
    with torch.no_grad():
        for i, data in enumerate(CLASSES):
            text_prompt = f"a photo of a {data}"
            text = clip.tokenize(text_prompt).to(device)
            text_features = model.encode_text(text).cpu().numpy()  # （1,512）
            one_sample = np.array(text_features)
            all_label_features[i, :] = one_sample
    return all_label_features
all_label_features=generate_label(CLASSES)
print(1)


import pickle

P = os.getcwd()
print(P)
with open(join(P, "data/iaprtc/label_prompt_features.pkl"), 'wb') as f:
    pickle.dump(all_label_features, f)

