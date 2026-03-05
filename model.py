import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from print import printPltGmm





class LableNet(nn.Module):
    def __init__(self, img_feat_len,code_len):
        super(LableNet, self).__init__()
        self.fc1 = nn.Linear( img_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):

        x = x.view(x.size(0), -1).float()
        feat1 = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)



class ImgNet(nn.Module):
    def __init__(self, img_feat_len,code_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear( img_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):

        x = x.view(x.size(0), -1).float()
        feat1 = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self,  txt_feat_len,code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)

        # self.init_weights()   #coco时屏蔽掉  25k nus时保留
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)



def P_clip(emb, k):

    k = min(max(k, 1), emb.size(0))
    topk_values, topk_indices = torch.topk(emb, k, dim=1)

    emb_k= torch.zeros_like(emb)
    emb_k.scatter_(1, topk_indices, topk_values)
    return emb_k



def corrected_label(self,sort_ids,corrected_labels,noise_label,img,txt, lable_poxy,device=0):

    img, txt, noise_label,lable_poxy = img.to(device), txt.to(device), noise_label.float().to(device),lable_poxy.to(device)

    img_F = F.normalize(img)
    text_F = F.normalize(txt)
    lable_poxy_F = F.normalize(lable_poxy)

    image_clip_pseudo = img_F.mm(lable_poxy_F.t())
    text_clip_pseudo = text_F.mm(lable_poxy_F.t())

    image_clip_pseudo = F.normalize(image_clip_pseudo)
    text_clip_pseudo = F.normalize(text_clip_pseudo)



    # 使用模型自身的预测分布来决定阈值（仅使用sort_ids指定的样本）
    selected_image_conf = image_clip_pseudo[sort_ids].flatten()
    selected_text_conf = text_clip_pseudo[sort_ids].flatten()

    # printPlt(selected_image_conf, selected_text_conf)

    left_image, right_image=distribution(selected_image_conf,self.config.L1,self.config.L2)
    left_text, right_text=distribution(selected_text_conf,self.config.L1,self.config.L2)

    false_positives = torch.zeros_like(noise_label, dtype=torch.bool)
    false_negatives = torch.zeros_like(noise_label, dtype=torch.bool)

    for i in sort_ids:  # 遍历每个样本
        #遍历需要修正的样本
        if self.excluded_samples[i] and self.sample_modification_count[i] < self.max_modifications:
            continue

        # 当前样本的原始标签和置信度
        curr_label = noise_label[i]
        # --- 情况1：检测1→0噪声（仅允许修正一个正标签）---
        if (sum(curr_label) > 0):  # 检查是否存在正标签
            # pos_candidates = (curr_label == 1) & (image_clip_pseudo[i] < pos_threshold_I) & (text_clip_pseudo[i] < pos_threshold_T)
            pos_candidates = (curr_label == 1) & (image_clip_pseudo[i] < left_image) & (text_clip_pseudo[i] < left_text)
            if pos_candidates.sum() > 0:
                # 选择最可疑的1个标签（双模态置信度之和最低）
                worst_pos_idx = torch.argmin(image_clip_pseudo[i][pos_candidates] + text_clip_pseudo[i][pos_candidates])
                false_positives[i, pos_candidates.nonzero()[worst_pos_idx]] = True
                record_modification(self,i)

        # --- 情况2：检测0→1噪声（仅允许修正一个负标签）---
        if sum(curr_label) < noise_label.size(1):  # 检查是否存在负标签
            # neg_candidates = (curr_label == 0) & (image_clip_pseudo[i] > neg_threshold_I) & (text_clip_pseudo[i] > neg_threshold_T)
            neg_candidates = (curr_label == 0) & (image_clip_pseudo[i] > right_image) & (text_clip_pseudo[i] >  right_text)
            if neg_candidates.sum() > 0:
                # 选择最可疑的1个标签（双模态置信度之和最高）
                worst_neg_idx = torch.argmax(image_clip_pseudo[i][neg_candidates] + text_clip_pseudo[i][neg_candidates])
                false_negatives[i, neg_candidates.nonzero()[worst_neg_idx]] = True
                record_modification(self, i)

    print("false_positives",torch.sum(false_positives))
    print("false_negatives", torch.sum(false_negatives))


    # corrected_labels[false_positives] = 0  # 将假阳性修正为0
    # corrected_labels[false_negatives] = 1  # 将假阴性修正为1

    for i in sort_ids:
        mod_count = self.sample_modification_count[i]

        weight = min(0 + (mod_count) * self.config.epsilon, 0.5)

        # 修正假阳性（仅修改对应位置的标签）
        fp_mask = false_positives[i]
        corrected_labels[i][fp_mask] = weight
        # 修正假阴性（仅修改对应位置的标签）
        fn_mask = false_negatives[i]
        corrected_labels[i][fn_mask] = 1-weight



def corrected_label2(self,noisy_ids, clean_ids,corrected_labels,noise_label,img,txt, lable_poxy,device=0):

    img, txt, noise_label,lable_poxy = img.to(device), txt.to(device), noise_label.float().to(device),lable_poxy.to(device)



    img=self.encode_image(img).detach()
    txt=self.encode_text(txt).detach()
    lable_poxy= self.encode_label(lable_poxy).detach()
    # lable_poxy= self.encode_text(lable_poxy).detach()

    img_F = F.normalize(img)
    text_F = F.normalize(txt)

    lable_poxy_F = F.normalize(lable_poxy)

    image_clip_pseudo = img_F.mm(lable_poxy_F.t())
    text_clip_pseudo = text_F.mm(lable_poxy_F.t())

    image_clip_pseudo = F.normalize(image_clip_pseudo)
    text_clip_pseudo = F.normalize(text_clip_pseudo)

    # image_clip_pseudo =torch.sigmoid(image_clip_pseudo )
    # text_clip_pseudo =torch.sigmoid(text_clip_pseudo)



    selected_image_conf = image_clip_pseudo[clean_ids]
    selected_text_conf = text_clip_pseudo[clean_ids]

    l1=noise_label[clean_ids]
    l0=(1-noise_label[clean_ids])

    left_image=selected_image_conf *l0

    left_image =torch.sum(left_image,dim=-1)/torch.sum(l0,dim=-1)
    left_image=torch.mean(left_image,dim=-1)


    right_image =selected_image_conf *l1

    right_image = torch.sum(right_image, dim=-1) / torch.sum(l1, dim=-1)
    right_image = torch.mean(right_image, dim=-1)




    left_text=selected_text_conf *l0

    left_text = torch.sum(left_text, dim=-1) / torch.sum(l0, dim=-1)
    left_text = torch.mean(left_text, dim=-1)


    right_text=selected_text_conf *l1

    right_text = torch.sum(right_text, dim=-1) / torch.sum(l1, dim=-1)
    right_text = torch.mean(right_text, dim=-1)




    printPlt((selected_image_conf * l1).flatten(), (selected_image_conf * l0).flatten())
    printPlt((selected_text_conf * l1).flatten(), (selected_text_conf * l0).flatten())

    false_positives = torch.zeros_like(noise_label, dtype=torch.bool)
    false_negatives = torch.zeros_like(noise_label, dtype=torch.bool)

    for i in noisy_ids:  # 遍历每个样本
        #遍历需要修正的样本
        if self.excluded_samples[i] and self.sample_modification_count[i] < self.max_modifications:
            continue

        # 当前样本的原始标签和置信度
        curr_label = noise_label[i]
        # --- 情况1：检测1→0噪声（假阳性）---
        if (sum(curr_label) > 0):  # 检查是否存在正标签
            # 分别检测图像模态和文本模态的异常
            pos_candidates_I = (curr_label == 1) & (image_clip_pseudo[i] < left_image)
            pos_candidates_T = (curr_label == 1) & (text_clip_pseudo[i] < left_text)

            # 合并两个模态的候选（取并集）
            pos_candidates = pos_candidates_I | pos_candidates_T

            if pos_candidates.sum() > 0:
                candidate_indices = pos_candidates.nonzero().squeeze(1)

                # 获取其他正标签作为参考
                # other_pos_mask = (curr_label == 1) & ~pos_candidates
                other_pos_mask = (curr_label == 1)

                if other_pos_mask.sum() > 0:
                    μ_I = image_clip_pseudo[i][other_pos_mask].mean()
                    μ_T = text_clip_pseudo[i][other_pos_mask].mean()

                # 计算每个候选标签的相对异常度
                relative_scores = []
                for j in candidate_indices:
                        # 分别计算图像和文本的异常度
                        rel_I = μ_I - image_clip_pseudo[i][j] if pos_candidates_I[j] else 0
                        rel_T = μ_T - text_clip_pseudo[i][j] if pos_candidates_T[j] else 0
                        total_anomaly = rel_I + rel_T
                        relative_scores.append(total_anomaly)

                relative_scores = torch.stack(relative_scores)

                # 选择最可疑的
                worst_pos_idx = torch.argmax(relative_scores)
                if worst_pos_idx < len(candidate_indices):
                        false_positives[i, candidate_indices[worst_pos_idx]] = True
                        record_modification(self, i)

        # --- 情况2：检测0→1噪声（假阴性）---
        if sum(curr_label) < noise_label.size(1):  # 检查是否存在负标签
            # 分别检测图像模态和文本模态的异常
            neg_candidates_I = (curr_label == 0) & (image_clip_pseudo[i] > right_image)
            neg_candidates_T = (curr_label == 0) & (text_clip_pseudo[i] > right_text)

            # 合并两个模态的候选（取并集）
            neg_candidates = neg_candidates_I | neg_candidates_T

            if neg_candidates.sum() > 0:
                candidate_indices = neg_candidates.nonzero().squeeze(1)

                # 获取其他负标签作为参考
                # other_neg_mask = (curr_label == 0) & ~neg_candidates
                other_neg_mask = (curr_label == 0)

                if other_neg_mask.sum() > 0:
                    μ_I = image_clip_pseudo[i][other_neg_mask].mean()
                    μ_T = text_clip_pseudo[i][other_neg_mask].mean()

                # 计算每个候选标签的相对异常度
                relative_scores = []
                for j in candidate_indices:
                    # 分别计算图像和文本的异常度
                    rel_I = image_clip_pseudo[i][j] - μ_I if neg_candidates_I[j] else 0
                    rel_T = text_clip_pseudo[i][j] - μ_T if neg_candidates_T[j] else 0
                    total_anomaly = rel_I + rel_T
                    relative_scores.append(total_anomaly)

                relative_scores = torch.stack(relative_scores)

                # 选择最可疑的
                worst_neg_idx = torch.argmax(relative_scores)
                if worst_neg_idx < len(candidate_indices):
                    false_negatives[i, candidate_indices[worst_neg_idx]] = True
                    record_modification(self, i)

    print("false_positives",torch.sum(false_positives))
    print("false_negatives", torch.sum(false_negatives))


    # corrected_labels[false_positives] = 0  # 将假阳性修正为0
    # corrected_labels[false_negatives] = 1  # 将假阴性修正为1

    for i in noisy_ids:
        mod_count = self.sample_modification_count[i]

        weight = min(0 + (mod_count) * self.config.epsilon, 0.5)

        # 修正假阳性（仅修改对应位置的标签）
        fp_mask = false_positives[i]
        corrected_labels[i][fp_mask] = weight
        # 修正假阴性（仅修改对应位置的标签）
        fn_mask = false_negatives[i]
        corrected_labels[i][fn_mask] = 1-weight



def distribution(elements,L1=0.7,L2=1):
    mean_value = elements.mean().item()  # 计算均值
    std_value = elements.std().item()  # 计算标准差
    # max_value = elements.max().item()  # 计算最大值
    # min_value = elements.min().item()  # 计算最小值
    left =  mean_value -std_value*L1
    right = mean_value +std_value*L2
    return left, right


def record_modification(self, sample_idx):
        """记录标签修改事件"""
        if self.sample_modification_count[sample_idx] >= self.max_modifications:
            self.excluded_samples[sample_idx] = True
            return False

        self.sample_modification_count[sample_idx] += 1
        # 检查是否达到最大修改次数
        if self.sample_modification_count[sample_idx] >= self.max_modifications:
            self.excluded_samples[sample_idx] = True
        return True


def printPlt(selected_image,selected_text):

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # 设置默认字体大小

    # 确保张量在 CPU 上并转为 NumPy
    selected_image_conf_np = selected_image.cpu().detach().numpy()  # i2t 置信度
    selected_text_conf_np = selected_text.cpu().detach().numpy()  # t2i 置信度

    # 过滤掉0值，只保留大于0的值
    selected_image_conf_np = selected_image_conf_np[selected_image_conf_np > 0]
    selected_text_conf_np  = selected_text_conf_np[selected_text_conf_np > 0]

    # 创建 1x2 的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    # 为子图1添加加粗边框
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # 子图1：Image-to-Text (i2t) 分布
    ax1.hist(
        selected_image_conf_np,
        bins=500,
        color='#aab911',
        alpha=0.7,
        density=True,
        # label='Histogram'  # 添加柱状图标签
    )
    # 计算并绘制平滑密度曲线
    kde = gaussian_kde(selected_image_conf_np)
    x_vals = np.linspace(min(selected_image_conf_np), max(selected_image_conf_np), 1000)
    ax1.plot(
        x_vals,
        kde(x_vals),
        color='#2b702f',
        linewidth=2,
        # label='I→T Density'  # 添加密度曲线标签
        label = '$l_{i,j}=1$'
    )
    # ax1.set_title('l=1', fontweight='bold')  # 标题加粗
    # ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Density')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(framealpha=0.9,fontsize=20)  # 添加图例，并设置透明度

    # 为子图2添加加粗边框
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # 子图2：Text-to-Image (t2i) 分布
    ax2.hist(
        selected_text_conf_np,
        bins=500,
        color='#f4a52e',
        alpha=0.7,
        density=True,
        # label='Histogram'  # 添加柱状图标签
    )
    # 计算并绘制平滑密度曲线
    kde = gaussian_kde(selected_text_conf_np)
    x_vals = np.linspace(min(selected_text_conf_np), max(selected_text_conf_np), 1000)
    ax2.plot(
        x_vals,
        kde(x_vals),
        color='#ed7471',
        linewidth=2,
        # label='T→I Density'  # 添加密度曲线标签
        label = '$l_{i,j}=0$'
    )
    # ax2.set_title('T→I Confidence Distribution', fontweight='bold')  # 标题加粗
    # ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Density')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(framealpha=0.9,fontsize=20)  # 添加图例，并设置透明度

    # 设置 x 轴主刻度步长为 0.2
    ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    ax2.xaxis.set_major_locator(MultipleLocator(0.2))

    # 调整布局并显示
    plt.tight_layout()
    plt.show()



def adj_w(img,txt,lable_poxy,k=5):

    img_F = F.normalize(img)
    text_F = F.normalize(txt)
    lable_poxy_F = F.normalize(lable_poxy)

    image_clip_pseudo = img_F.mm(lable_poxy_F.t())
    text_clip_pseudo = text_F.mm(lable_poxy_F.t())

    image_clip_pseudo = torch.softmax(image_clip_pseudo, dim=-1)
    text_clip_pseudo = torch.softmax(text_clip_pseudo, dim=-1)

    batch_size = image_clip_pseudo.size(0)


    adj = torch.zeros((batch_size, batch_size), device=image_clip_pseudo.device)
    for i in range(k):
        if k > 0:
            adjImg_i = P_clip(image_clip_pseudo, k)
            adjTxt_i = P_clip(text_clip_pseudo, k)

            adj_i = adjImg_i.mm(adjTxt_i.t())
            adj += adj_i / k
            k = k - 1
    scale = 1
    adj_diff = torch.sigmoid(scale * adj)

    # adj = torch.zeros((batch_size), device=image_clip_pseudo.device)
    # for i in range(k):
    #     if k > 0:
    #         adjImg_i = P_clip(image_clip_pseudo, k)
    #         adjTxt_i = P_clip(text_clip_pseudo, k)
    #         adj_i =adjImg_i*adjTxt_i
    #         # adj_i = adjImg_i.mm(adjTxt_i.t())
    #         adj_i=adj_i.sum(dim=-1)
    #         adj += adj_i / k
    #         k = k - 1
    # scale = 1
    # adj_diff = torch.sigmoid(scale * adj)

    return adj_diff


def adj_matrix(S):
    S = 1 / (1 + torch.exp(-S))
    S = S * 2 - 1
    return S


def dyn_threshld(epoch,total_epochs=100,start_value=0.65,lowest_value=0.6):
        # 计算抛物线参数
        mid = 0.5 * total_epochs
        a = (start_value - lowest_value) / (mid ** 2)
        threshld = a * (epoch - mid) ** 2 + lowest_value
        return threshld

def Gmm(self,lossData,epoch, total_epochs=100, start_value=0.8, lowest_value=0.7):
    lossData = lossData.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(lossData.cpu().numpy())
    prob = gmm.predict_proba(lossData.cpu().numpy())
    prob_max = prob[:, gmm.means_.argmax()]


    threshld = dyn_threshld(epoch, total_epochs, start_value, lowest_value)
    print("threshld：", threshld)
    printPltGmm(self, gmm, lossData, threshld, epoch)
    noisy_mask = prob_max > threshld
    sort_ids = torch.from_numpy(np.where(noisy_mask)[0])  # 合并操作
    return sort_ids


def Gmm2(self,lossData,epoch,threshld):
    lossData = lossData.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(lossData.cpu().numpy())
    prob = gmm.predict_proba(lossData.cpu().numpy())
    prob_max = prob[:, gmm.means_.argmax()]
    # threshld = dyn_threshld(epoch, total_epochs, start_value, lowest_value)

    print("threshld：", threshld)
    printPltGmm(self, gmm, lossData, threshld, epoch)
    noisy_mask = prob_max > threshld
    clean_mask = ~noisy_mask

    noisy_indices = torch.from_numpy(np.where(noisy_mask)[0])  # 合并操作
    clean_indices = torch.from_numpy(np.where(clean_mask)[0])
    return noisy_indices, clean_indices





#
# class Proj_Pure_MLP(nn.Module):
#     def __init__(self, in_features, out_features, middle_dim):
#         super(Proj_Pure_MLP, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.MLP = nn.Sequential(
#             nn.Linear(in_features, out_features),
#         )
#
#     def forward(self, input):
#         out = self.MLP(input)
#         return out
#
#
#
#
# class encode_text(nn.Module):
#     def __init__(self, device):
#         super(encode_text, self).__init__()
#         self.feature_dim=512
#         self.device = device
#         self.projs_text = nn.ModuleList()
#         self.projs_text.append(self.extend_item())
#
#     def extend_item(self):
#             return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self.device)
#     def forward(self, x, normalize: bool = False):
#         x = x.to(self.device)
#         text_features = [proj(x) for proj in self.projs_text]
#         text_features = torch.stack(text_features, dim=1)
#         text_feas = torch.sum(text_features, dim=1)  # [bs,dim]
#         return F.normalize(text_feas, dim=-1) if normalize else text_feas
#
#
# class encode_image(nn.Module):
#     def __init__(self, device):
#         self.device=device
#         self.feature_dim = 512
#         super(encode_image, self).__init__()
#         self.projs_img = nn.ModuleList()
#         self.projs_img.append(self.extend_item())
#
#     def extend_item(self):
#         return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self.device)
#     def forward(self, x, normalize: bool = False):
#         x = x.to(self.device)
#         img_features = [proj(x) for proj in self.projs_img]
#         img_features = torch.stack(img_features, dim=1)  # [bs,num_proj,dim]
#         image_feas = torch.sum(img_features, dim=1)  # [bs,dim]
#         return F.normalize(image_feas, dim=-1) if normalize else image_feas
#
#
# class encode_label(nn.Module):
#     def __init__(self, device):
#         super(encode_label, self).__init__()
#         self.feature_dim=512
#         self.device = device
#         self.projs_text = nn.ModuleList()
#         self.projs_text.append(self.extend_item())
#
#     def extend_item(self):
#             return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self.device)
#     def forward(self, x, normalize: bool = False):
#         x = x.to(self.device)
#         text_features = [proj(x) for proj in self.projs_text]
#         text_features = torch.stack(text_features, dim=1)
#         text_feas = torch.sum(text_features, dim=1)  # [bs,dim]
#         return F.normalize(text_feas, dim=-1) if normalize else text_feas

# class DeepProj_MLP(nn.Module):
#     def __init__(self, in_features, out_features, num_layers=1, hidden_dim=None):
#         super().__init__()
#
#         # assert num_layers in [1, 2], "只支持1层或2层"
#
#         if num_layers == 1:
#             # 单层：直接映射
#             self.mlp = nn.Linear(in_features, out_features)
#         else:
#             # 双层：输入->隐藏->输出
#             if hidden_dim is None:
#                 hidden_dim = (in_features + out_features) // 2
#             self.mlp = nn.Sequential(
#                 nn.Linear(in_features, hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, out_features)
#             )
#
#     def forward(self, x):
#         return self.mlp(x)


class Proj_Pure_MLP(nn.Module):
    def __init__(self, in_features, out_features, middle_dim=None):  # middle_dim设为可选
        super().__init__()  # 简化super调用
        self.MLP = nn.Linear(in_features, out_features)

    def forward(self, x):  # 使用更常见的x作为输入名
        return self.MLP(x)

class encode_text(nn.Module):
    def __init__(self, device, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        # 直接初始化，避免不必要的复杂结构
        self.proj = Proj_Pure_MLP(feature_dim, feature_dim, feature_dim).to(device)
        # self.proj = DeepProj_MLP(feature_dim, feature_dim, feature_dim).to(device)


    def forward(self, x, normalize: bool = False):
        x = x.to(self.device)
        text_feas = self.proj(x)  # 直接使用单个投影
        return F.normalize(text_feas, dim=-1) if normalize else text_feas

class encode_image(nn.Module):
    def __init__(self, device, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.proj = Proj_Pure_MLP(feature_dim, feature_dim, feature_dim).to(device)
        # self.proj = DeepProj_MLP(feature_dim, feature_dim, feature_dim).to(device)
    def forward(self, x, normalize: bool = False):
        x = x.to(self.device)
        image_feas = self.proj(x)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas

class encode_label(nn.Module):
    def __init__(self, device, feature_dim=512):
        super(encode_label, self).__init__()
        self.feature_dim=feature_dim
        self.device = device
        self.proj = Proj_Pure_MLP(feature_dim, feature_dim, feature_dim).to(device)
        # self.proj = DeepProj_MLP(feature_dim, feature_dim, feature_dim).to(device)
    def extend_item(self):
            return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self.device)
    def forward(self, x, normalize: bool = False):
        x = x.to(self.device)
        image_feas = self.proj(x)
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
