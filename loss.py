import torch
import torch.nn.functional as F
import torch.nn as nn 
#
class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.device= device

    def forward(self, emb_i, emb_j,adj=None):
        batch_size = emb_i.shape[0]
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(self.device)).float()
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        # adj_d= torch.diag(adj)
        # adj=torch.cat([adj_d, adj_d], dim=0)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        # loss = torch.sum(loss_partial*adj) / torch.sum(adj)

        return loss
#

# class ContrastiveLoss(nn.Module):
#     def __init__(self, device='cuda:0', temperature=0.5, pos_weight_scale=1.0, neg_weight_scale=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))
#         self.pos_weight_scale = pos_weight_scale  # 控制正对权重的影响
#         self.neg_weight_scale = neg_weight_scale  # 控制负对权重的影响
#         self.device = device
#
#     def forward(self, emb_i, emb_j, adj):
#         batch_size = emb_i.shape[0]
#
#         # Normalize embeddings
#         z_i = F.normalize(emb_i, dim=1)  # (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)  # (bs, dim)
#
#         # Concatenate embeddings (z_i followed by z_j)
#         representations = torch.cat([z_i, z_j], dim=0)  # (2*bs, dim)
#
#         # Compute similarity matrix (cosine similarity)
#         similarity_matrix = F.cosine_similarity(
#             representations.unsqueeze(1),
#             representations.unsqueeze(0),
#             dim=2
#         )  # (2*bs, 2*bs)
#
#         # Positive pairs: sim(z_i, z_j) and sim(z_j, z_i)
#         sim_ij = torch.diag(similarity_matrix, batch_size)  # (bs,)
#         sim_ji = torch.diag(similarity_matrix, -batch_size)  # (bs,)
#
#         # Weight positive pairs by adj (assuming adj[i,j] is the confidence for (i,j))
#         adj_diag = torch.diag(adj)  # (bs,)
#         weighted_sim_ij = sim_ij * (1.0 + self.pos_weight_scale * adj_diag)  # enhance positive similarity
#         weighted_sim_ji = sim_ji * (1.0 + self.pos_weight_scale * adj_diag)  # enhance positive similarity
#
#         positives = torch.cat([weighted_sim_ij, weighted_sim_ji], dim=0)  # (2*bs,)
#
#         # Compute negative weights (reduce penalty if adj[i,k] is high but they are negatives)
#         adj_expanded = torch.zeros_like(similarity_matrix)  # (2*bs, 2*bs)
#         adj_expanded[:batch_size, batch_size:] = adj  # adj[i,j] for (i,j)
#         adj_expanded[batch_size:, :batch_size] = adj.T  # adj[j,i] for (j,i)
#
#         # Negatives mask (all except diagonal and (i,j), (j,i) pairs)
#         negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#
#         # Adjust negative weights: higher adj[i,k] means less penalty (if they are negatives)
#         neg_weights = 1.0 - self.neg_weight_scale * adj_expanded * negatives_mask
#
#         # Compute denominator (weighted sum of exp(sim / T))
#         denominator = (neg_weights * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)  # (2*bs,)
#
#         # Compute numerator (exp(sim_pos / T))
#         nominator = torch.exp(positives / self.temperature)  # (2*bs,)
#
#         # Compute loss
#         loss = -torch.log(nominator / (denominator + 1e-8)).mean()
#
#         return loss









#
# class ContrastiveLoss(nn.Module):
#     def __init__(self, device='cuda:0', temperature=0.5, pos_weight_scale=1.0, neg_weight_scale=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))
#         self.pos_weight_scale = pos_weight_scale  # 控制正对权重的影响
#         self.neg_weight_scale = neg_weight_scale  # 控制负对权重的影响
#         self.device = device
#
#     def forward(self, emb_i, emb_j, adj=None):
#         batch_size = emb_i.shape[0]
#
#         # Normalize embeddings
#         z_i = F.normalize(emb_i, dim=1)  # (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)  # (bs, dim)
#
#         # Concatenate embeddings (z_i followed by z_j)
#         representations = torch.cat([z_i, z_j], dim=0)  # (2*bs, dim)
#
#         # Compute similarity matrix (cosine similarity)
#         similarity_matrix = F.cosine_similarity(
#             representations.unsqueeze(1),
#             representations.unsqueeze(0),
#             dim=2
#         )  # (2*bs, 2*bs)
#
#         # Positive pairs: sim(z_i, z_j) and sim(z_j, z_i)
#         sim_ij = torch.diag(similarity_matrix, batch_size)  # (bs,)
#         sim_ji = torch.diag(similarity_matrix, -batch_size)  # (bs,)
#
#         if adj is not None:
#             # Weight positive pairs by adj (assuming adj[i,j] is the confidence for (i,j))
#             adj_diag = torch.diag(adj)  # (bs,)
#             weighted_sim_ij = sim_ij * (1.0 + self.pos_weight_scale * adj_diag)  # enhance positive similarity
#             weighted_sim_ji = sim_ji * (1.0 + self.pos_weight_scale * adj_diag)  # enhance positive similarity
#
#             positives = torch.cat([weighted_sim_ij, weighted_sim_ji], dim=0)  # (2*bs,)
#
#             # Compute negative weights (reduce penalty if adj[i,k] is high but they are negatives)
#             adj_expanded = torch.zeros_like(similarity_matrix)  # (2*bs, 2*bs)
#             adj_expanded[:batch_size, batch_size:] = adj  # adj[i,j] for (i,j)
#             adj_expanded[batch_size:, :batch_size] = adj.T  # adj[j,i] for (j,i)
#
#             # Negatives mask (all except diagonal and (i,j), (j,i) pairs)
#             negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#
#             # Adjust negative weights: higher adj[i,k] means less penalty (if they are negatives)
#             neg_weights = 1.0 - self.neg_weight_scale * adj_expanded * negatives_mask
#
#             # Compute denominator (weighted sum of exp(sim / T))
#             denominator = (neg_weights * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)  # (2*bs,)
#         else:
#             # Standard contrastive loss (no adj weighting)
#             positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2*bs,)
#
#             # Negatives mask (all except diagonal and (i,j), (j,i) pairs)
#             negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#
#             # Compute denominator (sum of exp(sim / T))
#             denominator = (negatives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)  # (2*bs,)
#
#         # Compute numerator (exp(sim_pos / T))
#         nominator = torch.exp(positives / self.temperature)  # (2*bs,)
#
#         # Compute loss
#         loss = -torch.log(nominator / (denominator + 1e-8)).mean()
#
#         return loss
#
# #
# class ContrastiveLoss(nn.Module):
#     def __init__(self, device='cuda:0', temperature=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))
#         self.device = device
#
#     def forward(self, emb_i, emb_j, adj=None):
#         batch_size = emb_i.shape[0]
#
#         # Normalize embeddings
#         z_i = F.normalize(emb_i, dim=1)  # (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)  # (bs, dim)
#
#         # Concatenate embeddings (z_i followed by z_j)
#         representations = torch.cat([z_i, z_j], dim=0)  # (2*bs, dim)
#
#         # Compute similarity matrix (cosine similarity)
#         similarity_matrix = F.cosine_similarity(
#             representations.unsqueeze(1),
#             representations.unsqueeze(0),
#             dim=2
#         )  # (2*bs, 2*bs)
#
#         # Positive pairs: sim(z_i, z_j) and sim(z_j, z_i)
#         sim_ij = torch.diag(similarity_matrix, batch_size)  # (bs,)
#         sim_ji = torch.diag(similarity_matrix, -batch_size)  # (bs,)
#
#         if adj is not None:
#             # Weight positive pairs (non-symmetric adj)
#             adj_ij = adj.diag()  # adj[i,j] for (i,j) pairs
#             adj_ji = adj.T.diag()  # adj[j,i] for (j,i) pairs
#             # weighted_sim_ij = sim_ij * (1.0 + adj_ij)  # 加权 sim(i,j)
#             # weighted_sim_ji = sim_ji * (1.0 + adj_ji)  # 加权 sim(j,i)
#             weighted_sim_ij = sim_ij * (adj_ij)  # 加权 sim(i,j)
#             weighted_sim_ji = sim_ji * (adj_ji)  # 加权 sim(j,i)
#
#             positives = torch.cat([weighted_sim_ij, weighted_sim_ji], dim=0)  # (2*bs,)
#
#             # Weight negative pairs (reduce penalty if adj[i,k] is high)
#             adj_expanded = torch.ones_like(similarity_matrix)  # (2*bs, 2*bs)
#             adj_expanded[:batch_size, batch_size:] = adj  # adj[i,j] for (i,j)
#             adj_expanded[batch_size:, :batch_size] = adj.T  # adj[j,i] for (j,i)
#
#             # neg_weights = 1.0 - adj_expanded  # Penalty reduction for high adj[i,k]
#             neg_weights = adj_expanded  # Penalty reduction for high adj[i,k]
#             # Negatives mask (exclude diagonal and positive pairs)
#             negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#
#             # Compute denominator (weighted sum of exp(sim / T))
#             denominator = (neg_weights * negatives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)
#         else:
#             # Standard contrastive loss (no adj weighting)
#             positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2*bs,)
#             negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#             denominator = (negatives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)
#
#         # Compute loss
#         nominator = torch.exp(positives / self.temperature)
#         loss = -torch.log(nominator / denominator).mean()
#         return loss

#
# class ContrastiveLoss(nn.Module):
#     def __init__(self, device='cuda:0', temperature=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))
#         self.device = device
#
#     def forward(self, emb_i, emb_j, adj=None):
#         batch_size = emb_i.shape[0]
#
#         # Normalize embeddings
#         z_i = F.normalize(emb_i, dim=1)  # (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)  # (bs, dim)
#
#         # Concatenate embeddings (z_i followed by z_j)
#         representations = torch.cat([z_i, z_j], dim=0)  # (2*bs, dim)
#
#         # Compute similarity matrix (cosine similarity)
#         similarity_matrix = F.cosine_similarity(
#             representations.unsqueeze(1),
#             representations.unsqueeze(0),
#             dim=2
#         )  # (2*bs, 2*bs)
#
#         # Positive pairs: sim(z_i, z_j) and sim(z_j, z_i)
#         sim_ij = torch.diag(similarity_matrix, batch_size)  # (bs,) 即 sim(img_i, text_i)
#         sim_ji = torch.diag(similarity_matrix, -batch_size)  # (bs,) 即 sim(text_i, img_i)
#
#         if adj is not None:
#             ##############################################
#             # 关键修改1：非对称正对加权
#             ##############################################
#             # 直接使用adj的对角线元素（即adj[i,i]）作为正对权重
#             adj_weights = torch.diag(adj)  # (bs,)
#
#             # 加权正样本相似度
#             weighted_sim_ij = sim_ij * (1.0 + adj_weights)  # 增强img_i和text_i的相似度
#             weighted_sim_ji = sim_ji * (1.0 + adj_weights)  # 增强text_i和img_i的相似度
#
#             positives = torch.cat([weighted_sim_ij, weighted_sim_ji], dim=0)  # (2*bs,)
#
#             ##############################################
#             # 关键修改2：非对称负对惩罚
#             ##############################################
#             # 构建完整的adj矩阵（考虑跨模态关系）
#             adj_expanded = torch.zeros_like(similarity_matrix)  # (2*bs, 2*bs)
#
#             # 左上块：img-text相似度（来自原始adj）
#             adj_expanded[:batch_size, batch_size:] = adj
#
#             # 左下块：text-img相似度（来自adj转置）
#             adj_expanded[batch_size:, :batch_size] = adj.T
#
#             # 负样本掩码（排除对角线和正对）
#             negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#
#             # 负样本惩罚权重（adj越高，惩罚越小）
#             # neg_weights = 1.0 - adj_expanded * negatives_mask
#             # 负样本惩罚权重（adj越高，惩罚越大）
#             neg_weights = 1.0 + adj_expanded * negatives_mask
#             # neg_weights = negatives_mask
#             denominator = (neg_weights * negatives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)
#             # denominator = ( negatives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)
#
#         else:
#             # 标准对比损失
#             positives = torch.cat([sim_ij, sim_ji], dim=0)
#             negatives_mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float().to(self.device)
#             denominator = (negatives_mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)
#
#         # 计算最终损失
#         nominator = torch.exp(positives / self.temperature)
#         loss = -torch.log(nominator / denominator ).mean()
#         # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
#         # loss = torch.sum(loss_partial) / (2 * batch_size)
#
#         return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, device='cuda:0', temperature=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))
#         self.device = device
#
#     def forward(self, emb_i, emb_j, adj=None):
#         batch_size = emb_i.shape[0]
#
#         # 创建负样本掩码（对角线为0，其余为1）
#         negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(self.device)).float()
#
#         # 对特征向量进行归一化
#         z_i = F.normalize(emb_i, dim=1)  # (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)  # (bs, dim)
#
#         # 拼接所有样本表示
#         representations = torch.cat([z_i, z_j], dim=0)  # (2*bs, dim)
#
#         # 计算余弦相似度矩阵
#         similarity_matrix = F.cosine_similarity(
#             representations.unsqueeze(1),
#             representations.unsqueeze(0),
#             dim=2
#         )  # (2*bs, 2*bs)
#
#         # 获取正样本对的相似度
#         sim_ij = torch.diag(similarity_matrix, batch_size)  # (bs,) - z_i与z_j的相似度
#         sim_ji = torch.diag(similarity_matrix, -batch_size)  # (bs,) - z_j与z_i的相似度
#
#         # 合并所有正样本对
#         positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2*bs,)
#
#         # 计算对比损失分子
#         nominator = torch.exp(positives / self.temperature)  # (2*bs,)
#
#         # 如果提供了邻接矩阵，则在分子上应用权重
#         if adj is not None:
#             adj = torch.cat([adj, adj], dim=0)  # 将adj扩展到匹配样本数量 (2*bs,)
#             nominator = nominator * adj.detach()  # 在分子上应用adj权重
#             # 关键修改：使用adj权重的和进行归一化
#             normalization_term = torch.sum(adj)
#         else:
#             normalization_term = 2 * batch_size
#         # 计算分母
#         denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)  # (2*bs, 2*bs)
#
#         # 计算损失
#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
#         loss = torch.sum(loss_partial) / normalization_term
#
#         return loss




def loss_w(H,S,W=None):
    if W is None:
        loss=(H - S.detach()).pow(2)
    else:
        loss=(W*(H-S.detach()).pow(2))
    totaloss=loss.mean()

    loss_data=loss.detach()
    mask = torch.eye(loss_data.size(0), dtype=torch.bool, device=loss_data.device)  # 生成对角线为 True 的布尔矩阵
    # 将对角线元素置为 0
    loss_data[mask] = 0
    loss_data=loss_data.mean(dim=-1)

    # 将对角线外元素置为 0
    # loss_data[~mask] = 0
    # loss_data = loss_data.sum(dim=-1)

    # 标准化
    # loss_mean = loss_data.mean()
    # loss_std = loss_data.std()
    # loss_data= (loss_data - loss_mean) / (loss_std + 1e-8)  # 避免除以0

    #归一化 将所有值线性映射到 [0, 1] 区间
    # loss_data = (loss_data - loss_data.min()) / (loss_data.max() - loss_data.min())
    # loss_data=F.normalize(loss_data.squeeze(0),dim=-1)

    # soft归一化
    # loss_data =torch.softmax(loss_data, dim=-1)

    return totaloss,loss_data



def loss_w2(H,S,W):
    loss = W * (H - S.detach()).pow(2)
    num=torch.sum(W)
    if num>0:
        totaloss=loss.sum()/num
        return totaloss
    else:
        return 0


    # S=(S>0).float()
    # H=H*0.5
    # loss = (S * H - torch.log(1 + torch.exp(H)))
    # totaloss=-torch.mean( (S * H - torch.log(1 + torch.exp(H))))

    # loss = (H - S.detach()).pow(2)
    # totaloss = torch.mean(loss )
    # return totaloss

class SampleLossStorage:
    def __init__(self, num_samples):
        # [4, num_samples] 存储最近4轮每个样本的损失
        self.sample_loss_window = torch.zeros((4, num_samples))
        self.current_idx = 0  # 当前写入位置
        self.window_filled = False  # 窗口是否已填满4轮

    def add_epoch(self, sample_losses):
        """添加当前轮次的样本损失"""
        # 存储当前轮次
        self.sample_loss_window[self.current_idx] = sample_losses.detach().clone()

        # 更新索引
        self.current_idx = (self.current_idx + 1) % 4
        if self.current_idx == 0:
            self.window_filled = True

    def get_current_window(self):
        """获取当前4轮窗口（按时间顺序从旧到新）"""
        if not self.window_filled:
            return self.sample_loss_window[:self.current_idx]

        # 重新排序为时间顺序
        indices = [(self.current_idx + i) % 4 for i in range(4)]
        return self.sample_loss_window[indices]




#
# def loss_cra(embedding_1, embedding_2, C_L, corrected_labels, clean_mask):
#     batch_size = embedding_1.size(0)
#     device = embedding_1.device
#
#     # 提前转换数据类型和设备
#     corrected_labels = corrected_labels.to(device, dtype=torch.float32)
#     clean_mask = clean_mask.to(device, dtype=torch.float32)
#
#     # 计算相似度矩阵
#     E_C = embedding_1.mm(C_L.t())
#
#     # 创建正负样本掩码
#     l1 = corrected_labels  # 正样本
#     l0 = 1 - corrected_labels  # 负样本
#
#     # 数值稳定的指数计算
#     # 减去最大值提高数值稳定性
#     E_C_max = E_C.max(dim=1, keepdim=True)[0].detach()
#     E_C_stable = E_C - E_C_max
#
#     exp_E_C = torch.exp(E_C_stable)
#
#     # 计算正负项
#     pos = exp_E_C * l1
#     neg = exp_E_C * l0
#
#     # 计算分母（负样本和）
#     sum_neg = torch.sum(neg, dim=1, keepdim=True)
#
#     # 创建有效样本掩码（至少有一个正例且clean_mask为True）
#     valid_mask = (torch.sum(l1, dim=1) > 0) & (clean_mask > 0)
#     valid_count = torch.sum(valid_mask.float())
#
#     # 计算比率
#     ratio = pos / (sum_neg + 1e-12)  # 使用更小的epsilon
#
#     # 对每个样本的正例比率求和
#     ratio_sum = torch.sum(ratio, dim=1)
#
#     # 计算损失（只对有效样本）
#     loss_partial = -torch.log(ratio_sum + 1e-12)
#     loss_partial = loss_partial * valid_mask.float()
#
#     # 计算正则化项（正样本数量）
#     pos_count = torch.sum(l1)  # 总正样本数
#
#     # 最终损失计算
#     if valid_count > 0:
#         loss = torch.sum(loss_partial) / (valid_count + pos_count)
#     else:
#         loss = torch.tensor(0.0, device=device)
#
#     return loss











def loss_info(embedding_1, embedding_2, C_L, corrected_labels, clean_mask, temperature=0.1):
    """
    基于InfoNCE的对比损失，更稳定且理论基础更好
    """
    batch_size = embedding_1.size(0)
    device = embedding_1.device

    corrected_labels = corrected_labels.to(device, dtype=torch.float32)
    clean_mask = clean_mask.to(device, dtype=torch.float32)

    # 计算相似度矩阵
    similarity = embedding_1.mm(C_L.t()) / temperature

    # 创建正样本掩码
    pos_mask = corrected_labels.bool()

    # 有效样本掩码
    t1= (pos_mask.sum(dim=1) > 0)
    t2= clean_mask.bool().squeeze(dim=-1)

    valid_mask = t1 & t2

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # 只处理有效样本
    similarity_valid = similarity[valid_mask]
    pos_mask_valid = pos_mask[valid_mask]

    # 计算对比损失
    exp_sim = torch.exp(similarity_valid)

    # 正样本分数
    pos_scores = torch.sum(exp_sim * pos_mask_valid.float(), dim=1)

    # 所有样本分数（包括正负）
    all_scores = torch.sum(exp_sim, dim=1)

    # InfoNCE损失
    loss = -torch.log(pos_scores / (all_scores + 1e-12))

    return loss.mean()


