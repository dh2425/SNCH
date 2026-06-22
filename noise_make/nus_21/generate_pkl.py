import pickle
import numpy as np
from tqdm import tqdm




def add_noise_to_labels(labels, noise_rate):
    num_samples, num_labels = labels.shape
    num_noise = int(num_samples * noise_rate)
    # 创建一个随机的索引列表，用于选择要添加噪声的样本
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    # 随机改变选定样本的标签
    for i in tqdm(noise_indices):
        ones_indices = np.where(labels[i, :] == 1)[0]
        zeros_indices = np.where(labels[i, :] == 0)[0]
        # 随机选择一个值为1的元素，并将其变为0
        if len(ones_indices) > 0:
            j = np.random.choice(ones_indices)
            labels[i, j] = 0

        # 随机选择一个值为0的元素，并将其变为1
        if len(zeros_indices) > 0:
            j = np.random.choice(zeros_indices)
            labels[i, j] = 1
    return labels


def generate_noise_F(noise,all_loc):
    noise_rate = noise

    with open(all_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)

    for i in noise_rate:
        # 添加噪声
        labels_matrix = np.array(list(data['label']))
        labels_matrix2 = np.array(list(data['label']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)
        # 创建要保存的字典
        output_data = {
            'result': noisy_labels_matrix,
            'True': labels_matrix2
        }
        # 保存为pkl文件
        with open('noise/nus_21-lall-noise_21_{}.pkl'.format(i), 'wb') as f:
            pickle.dump(output_data, f)


all_loc = r'E:\PapperProgram\dataset_noise\clip_feature\nus\feature\train.pkl'
noise_rate = [0.2, 0.5, 0.8]
generate_noise_F(noise_rate,all_loc)
