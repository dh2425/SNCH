import json
import os
import numpy as np
from os.path import join


img_path = 'all/images.txt'
txt_path = 'all/texts.txt'
lab_path = 'all/labels.txt'

with open(img_path, 'r', encoding='utf-8') as f:
    img_datas =np.array([line.strip() for line in f])
    # img_datas = f.readlines()

with open(txt_path, 'r', encoding='utf-8') as f:
    txt_datas  = np.array([line.strip() for line in f])
    # txt_datas  = f.readlines()

lab_datas  = np.loadtxt(lab_path, dtype=np.int32)


all_image=img_datas
all_text=txt_datas
all_lab=lab_datas

query_num=2000
train_num=5000

random_index=np.random.permutation(range(len(all_image)))

query_index=random_index[:query_num]
train_index=random_index[query_num:query_num+train_num]
retrieval_index=random_index[query_num:]





# 创建保存数据的函数
def save_dataset(data_type, img_data, txt_data, label_data):
    # 创建目录
    os.makedirs(data_type, exist_ok=True)

    # 保存图片路径
    with open(join(data_type, 'images.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(img_data))

    # 保存文本数据
    with open(join(data_type, 'texts.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_data))

    # 保存标签数据
    with open(join(data_type, 'labels.txt'), 'w', encoding='utf-8') as f:
        for label in label_data:
            f.write(' '.join(map(str, label)))  # 将numpy数组转为字符串
            f.write('\n')
#
# 获取当前工作目录
P = os.getcwd()
print(f"当前工作目录: {P}")

# 保存数据集
save_dataset(join(P, 'train'),
             img_datas[train_index],
             txt_datas[train_index],
             lab_datas[train_index])

save_dataset(join(P, 'query'),
             img_datas[query_index],
             txt_datas[query_index],
             lab_datas[query_index])

save_dataset(join(P, 'retrieval'),
             img_datas[retrieval_index],
             txt_datas[retrieval_index],
             lab_datas[retrieval_index])

print("数据已成功保存到对应文件夹！")