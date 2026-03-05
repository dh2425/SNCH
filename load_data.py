import os

from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import torch

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs,noise_label=None):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.noise_label = noise_label
    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        if self.noise_label is not None :
            noise_label=self.noise_label[index]
            return img, text, lab, index, noise_label
        else:
            return img, text, lab, index

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataset(dataset,data_pth, batch_size):
    '''
        load datasets : flickr25k, mscoco, nus-wide
    '''

    # file_path = os.path.join(data_pth, dataset)
    """25K"""
    train_loc =r"E:\PapperProgram\dataset_noise\trans\flickr\after\train_25k.pkl"
    query_loc =r"E:\PapperProgram\dataset_noise\trans\flickr\after\query_25k.pkl"
    retrieval_loc =r"E:\PapperProgram\dataset_noise\trans\flickr\after\eval_25k.pkl"
    # noise = r"E:\PapperProgram\dataset_noise\noise_make\flickr\noise\mirflickr25k-lall-noise_0.2.pkl"
    noise=r"E:\PapperProgram\dataset_noise\noise_make\flickr\noise\mirflickr25k-lall-noise_0.5.pkl"
    # noise = r"E:\PapperProgram\dataset_noise\noise_make\flickr\noise\mirflickr25k-lall-noise_0.8.pkl"
    lable_poxy = r"E:\PapperProgram\dataset_noise\clip_lable_poxy\promot\data\25k\label_prompt_features.pkl"


    """COCO"""
    train_loc =r"E:\PapperProgram\dataset_noise\trans\coco\after\train_coco.pkl"
    query_loc =r"E:\PapperProgram\dataset_noise\trans\coco\after\query_coco.pkl"
    retrieval_loc =r"E:\PapperProgram\dataset_noise\trans\coco\after\eval_coco.pkl"
    noise=r"E:\PapperProgram\dataset_noise\noise_make\coco\noise\coco-lall-noise_0.2.pkl"
    # noise=r"E:\PapperProgram\dataset_noise\noise_make\coco\noise\coco-lall-noise_0.5.pkl"
    # noise = r"E:\PapperProgram\dataset_noise\noise_make\coco\noise\coco-lall-noise_0.8.pkl"
    lable_poxy =  r"E:\PapperProgram\dataset_noise\clip_lable_poxy\promot\data\coco\label_prompt_features.pkl"



    # """nus"""
    # train_loc =r"E:\PapperProgram\dataset_noise\trans\nus\after\train_nus.pkl"
    # query_loc =r"E:\PapperProgram\dataset_noise\trans\nus\after\query_nus.pkl"
    # retrieval_loc =r"E:\PapperProgram\dataset_noise\trans\nus\after\eval_nus.pkl"
    # # noise = r"E:\PapperProgram\dataset_noise\noise_make\nus\noise\nus-lall-noise_0.2.pkl"
    # # noise=r"E:\PapperProgram\dataset_noise\noise_make\nus\noise\nus-lall-noise_0.5.pkl"
    # noise = r"E:\PapperProgram\dataset_noise\noise_make\nus\noise\nus-lall-noise_0.8.pkl"
    # lable_poxy = r"E:\PapperProgram\dataset_noise\clip_lable_poxy\promot\data\nus\label_prompt_features.pkl"


    # """nus—21"""
    # train_loc =r"E:\PapperProgram\dataset_noise\clip_feature\nus_21\feature\train.pkl"
    # query_loc =r"E:\PapperProgram\dataset_noise\clip_feature\nus_21\feature\query.pkl"
    # retrieval_loc =r"E:\PapperProgram\dataset_noise\clip_feature\nus_21\feature\retrival.pkl"
    # noise = r"E:\PapperProgram\dataset_noise\noise_make\nus_21\noise\nus-lall-noise_21_0.5.pkl"
    # lable_poxy = r"E:\PapperProgram\dataset_noise\clip_lable_poxy\data\nus\label_features_21.pkl"
    # lable_poxy = r"E:\PapperProgram\dataset_noise\clip_lable_poxy\promot\data\nus\label_prompt_features_21.pkl"


    # """iaprtc"""
    train_loc =r"E:\PapperProgram\dataset_noise\clip_feature\iaprtc\feature\train.pkl"
    query_loc =r"E:\PapperProgram\dataset_noise\clip_feature\iaprtc\feature\query.pkl"
    retrieval_loc =r"E:\PapperProgram\dataset_noise\clip_feature\iaprtc\feature\retrival.pkl"
    # noise = r"E:\PapperProgram\dataset_noise\noise_make\iaprtc\noise\iaprtc-lall-noise_21_0.5.pkl"
    noise = r"E:\PapperProgram\dataset_noise\noise_make\iaprtc\noise\iaprtc-lall-noise_21_0.2.pkl"
    lable_poxy = r"E:\PapperProgram\dataset_noise\clip_lable_poxy\promot\data\iaprtc\label_prompt_features.pkl"

    with open(noise, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        noise_label = torch.tensor(data['result'], dtype=torch.int64)
        # real_label = torch.tensor(data['True'], dtype=torch.float32)

    with open(lable_poxy, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        lable_poxy = torch.tensor(data, dtype=torch.float32)

    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = torch.tensor(data['label'],dtype=torch.int64)
        train_texts = torch.tensor(data['text'], dtype=torch.float32)
        train_images = torch.tensor(data['image'], dtype=torch.float32)

    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_labels = torch.tensor(data['label'], dtype=torch.int64)
        query_texts = torch.tensor(data['text'], dtype=torch.float32)
        query_images = torch.tensor(data['image'], dtype=torch.float32)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_lables = torch.tensor(data['label'], dtype=torch.int64)
        retrieval_texts = torch.tensor(data['text'], dtype=torch.float32)
        retrieval_images = torch.tensor(data['image'], dtype=torch.float32)

    dataset_train =CustomDataSet(images=train_images, texts=train_texts, labs=train_labels,noise_label=noise_label)
    dataset_query = CustomDataSet(images=query_images, texts=query_texts, labs=query_labels)
    dataset_retrival = CustomDataSet(images=retrieval_images, texts=retrieval_texts, labs=retrieval_lables)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,shuffle=True, num_workers=0)
    dataloader_query = DataLoader(dataset_query, batch_size=batch_size,  drop_last=True, pin_memory=True, shuffle=False, num_workers=0)
    dataloader_retrival = DataLoader(dataset_retrival, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=False,num_workers=0)


    return dataloader_train ,dataloader_query,dataloader_retrival,lable_poxy

