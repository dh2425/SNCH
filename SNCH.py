
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from evaluate import calc_map_k
from load_data import  load_dataset
from loss import ContrastiveLoss,loss_w, loss_info
from model import ImgNet,TxtNet,adj_matrix, corrected_label2,Gmm2,encode_text,encode_image,encode_label
from utils import load_checkpoints, save_checkpoints, save_mat,save_feature
from torch.optim import lr_scheduler
import datetime
from collections import deque
from sklearn.mixture import GaussianMixture
from save import save_loss_realtime,save_percentage, save_double_loss_realtime,log_map_results,all_flue,print_lossData,save_double_loss_epoch
from print import printPltGmm


class SNCH(object):
    def __init__(self, log,config):
        self.logger=log
        self.config=config
        self.dataset = config.dataset
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.nbits = config.hash_lens

        self._init_dataset()
        self._init_model()
        self.max_map = {'i2t': 0, "t2i": 0}
        self.best_epoch=0




    def _init_dataset(self):
        dataloader_train, dataloader_query, dataloader_retrival ,lable_poxy= load_dataset(self.config.dataset, self.config.data_pth,
                                                                   self.config.batch_size)
        self.train_loader = dataloader_train
        self.query_loader = dataloader_query
        self.retrieval_loader =  dataloader_retrival
        self.lable_poxy=lable_poxy.float().to(self.device)


    def _init_model(self):
        self.log_filename = self._generate_log()
        self.step_counts=0


        # hash layer
        self.ImageMlp = ImgNet(self.config.feat_lens, self.config.hash_lens).to(self.device)
        self.TextMlp = TxtNet(self.config.feat_lens, self.config.hash_lens).to(self.device)

        self.encode_text = encode_text(self.device).to(self.device)
        self.encode_image =  encode_image(self.device).to(self.device)
        self.encode_label = encode_label(self.device).to(self.device)


        #params
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())

        paramsEncode_text= list(self.encode_text.parameters())
        paramsEncode_image = list(self.encode_image.parameters())
        paramsEncode_label= list(self.encode_label.parameters())

        # total_param=(sum([param.nelement() for param in paramsImage])
        #              +sum([param.nelement() for param in paramsText]))
        # print('Total number of parameters: {}'.format(total_param))

        self.optimizer_ImageMlp = optim.AdamW(paramsImage, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.AdamW(paramsText, lr=1e-3, betas=(0.5, 0.999))

        self.optimizer_Encode_text = optim.AdamW(paramsEncode_text, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_Encode_image= optim.AdamW(paramsEncode_image, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_Encode_label = optim.AdamW(paramsEncode_label, lr=1e-5, betas=(0.5, 0.999))


        scheduler_params = {
            "flickr25k": {"milestones": [120, 320], "gamma": 1.2},
            "nus-wide": {"milestones": [30, 80], "gamma": 1.2},
            "mscoco": {"milestones": [200], "gamma": 0.6},
        }
        params = scheduler_params.get(self.dataset)

        if params:
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, **params)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, **params)

            self.LableMlp_Encode_text_scheduler = lr_scheduler.MultiStepLR(self.optimizer_Encode_text, **params)
            self.LableMlp_Encode_image_scheduler = lr_scheduler.MultiStepLR(self.optimizer_Encode_image, **params)
            self.LableMlp_Encode_label_scheduler = lr_scheduler.MultiStepLR(self.optimizer_Encode_label, **params)


        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        self.ContrastiveLoss = ContrastiveLoss(device=self.device)

        train_samples=len(self.train_loader.dataset)
        self.lossData = torch.zeros(train_samples)
        self.lossData_nor =torch.zeros(train_samples)
        self.lossData_previous = torch.zeros(train_samples)
        self.corrected_labels= torch.as_tensor(self.train_loader.dataset.noise_label.clone(),dtype=torch.float32)


        #======================================
        self.noise_label = torch.as_tensor(self.train_loader.dataset.noise_label, dtype=torch.float32)
        self.real_label =torch.as_tensor(self.train_loader.dataset.labs, dtype=torch.float32)
        # # 假设 c1 和 c2 是形状为 (5000, 24) 的张量
        diff_mask = ~torch.all(torch.eq(self.noise_label, self.real_label ), dim=1)  # 检查每行是否全部相等
        self.noise_index = torch.where(diff_mask)[0]
        self.real_index = torch.where(~diff_mask)[0] # 取反，True 表示非噪声
        #======================================
        # 记录每个样本的总修改次数(跨所有类别)
        self.clean_mask =torch.ones(5000, dtype=torch.bool)

        self.sample_modification_count = torch.zeros(train_samples, dtype=torch.int)
        self.max_modifications = self.config.max_modify
        self.excluded_samples = torch.zeros(train_samples, dtype=torch.bool)
        self.lossData_history = deque([torch.tensor(0.0)] * (self.config.T + 1), maxlen=self.config.T+1)



    def _generate_log(self):
        """生成带时间戳的日志文件名"""
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        return current_time


    def train_epoch(self,epoch):
        running_loss = 0.
        self.ImageMlp.train()
        self.TextMlp.train()

        self.lossData= (self.lossData - self.lossData.min()) / (self.lossData.max() - self.lossData.min()+1e-10)

        if (epoch + 1) > self.config.warmup and (epoch + 1) % (self.config.T + 1) == 0:

            beta = torch.exp(torch.tensor(-self.config.α / self.config.T))
            weight = torch.pow(beta, torch.arange(0, self.config.T, dtype=torch.float32))
            weight = weight / weight.sum()  # 归一化
            all_loss = 0
            # 计算所有时间步的梯度变化
            gradient_changes = []
            for i in range(1, self.config.T + 1):
                # 梯度变化是按时间倒序排列的，从最近的时间段到最早的时间段
                # 对应时间: [t_{n+1}→t_{n}, t_{n}→t_n-1, ..., t_2→t_1, t_1→t_0]
                grad_change = self.lossData_history[i] - self.lossData_history[i - 1]

                # 正序存储: [t_{n+1}→t_{n}, t_{n}→t_n-1, ..., t_2→t_1, t_1→t_0]
                #gradient_changes.append(torch.abs(grad_change))
                # 逆序存储: [t_0→t_1，t_1→t_2, ...,t_{n-1}→t_n,t_{n}→t_{n+1}]
                gradient_changes.insert(0, torch.abs(grad_change))
            for i, grad_change in enumerate(gradient_changes):
                if i < len(weight):
                    all_loss += weight[i] * grad_change

            noisy_ids_fluent, clean_ids_fluent = Gmm2(self, all_loss, epoch,self.config.threshld )
            noisy_ids_last, clean_ids_last = Gmm2(self, self.lossData, epoch,self.config.threshld)

            # 并集
            noisy_ids_union = torch.cat([noisy_ids_fluent, noisy_ids_last])
            noisy_ids = torch.unique(noisy_ids_union)
            # 交集
            common_mask = torch.isin(clean_ids_fluent, clean_ids_last)
            clean_ids = clean_ids_fluent[common_mask]
            self.clean_mask[clean_ids] = True

            printlog(self, epoch, noisy_ids)
            percentage = torch.isin(noisy_ids, self.noise_index).sum().item() / (len(noisy_ids)) * 100
            save_percentage(self, epoch, percentage)

            corrected_label2(self, noisy_ids, clean_ids, self.corrected_labels,
                             torch.as_tensor(self.train_loader.dataset.noise_label, dtype=torch.float32),
                             torch.as_tensor(self.train_loader.dataset.images, dtype=torch.float32),
                             torch.as_tensor(self.train_loader.dataset.texts, dtype=torch.float32),
                             self.lable_poxy)

        S_all = self.corrected_labels.mm(self.corrected_labels.t())
        S_mask = ~torch.outer(self.excluded_samples, self.excluded_samples)  # 保持布尔型
        self.lossData_previous=self.lossData.clone()
        self.lossData_history.append(self.lossData.clone())

        for idx, (img, txt, labl,index, noise_label) in enumerate(self.train_loader):
            img, txt ,labl, noise_label= img.to(self.device), txt.to(self.device),labl.float().to(self.device),noise_label.float().to(self.device)

            img_code = self.ImageMlp(img)
            text_code = self.TextMlp(txt)

            S = S_all[index, :][:, index].cuda()
            Mask= S_mask[index, :][:, index].cuda()
            c_m=self.clean_mask[index].cuda().float().unsqueeze(1)
            clean_mask=c_m.bool()
            S_adj=adj_matrix(S)

            corrected_labels=self.corrected_labels[index]
            encode_img = self.encode_image(img)
            encode_txt = self.encode_text(txt)
            lable_poxy = self.encode_label(self.lable_poxy)
            loss,loss_s_data,loss_s= self.train_loss(img_code, text_code,encode_img,encode_txt ,S_adj,Mask,clean_mask,lable_poxy,corrected_labels)



            for i in range(img.size(0)):
                self.lossData[index[i]]=loss_s_data[i]

            if epoch>(self.lossData_history.maxlen+1):
                save_double_loss_realtime(self,index, img.size(0),self.step_counts,self.lossData_history,self.lossData)
            save_loss_realtime(self,loss_s.item())

            self.step_counts+=1
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            self.optimizer_Encode_text.zero_grad()
            self.optimizer_Encode_image.zero_grad()
            self.optimizer_Encode_label.zero_grad()
            loss.backward()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            self.optimizer_Encode_text.step()
            self.optimizer_Encode_image.step()
            self.optimizer_Encode_label.step()
            running_loss += loss.item()
            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()
            self.LableMlp_Encode_text_scheduler.step()
            self.LableMlp_Encode_image_scheduler.step()
            self.LableMlp_Encode_label_scheduler.step()
        return running_loss




    def train_loss(self,img_embedding,text_embedding,encode_img,encode_txt,S,S_mask=None,clean_mask=None,lable_poxy=None,corrected_labels=None):


        img_E = F.normalize(encode_img)
        text_E = F.normalize(encode_txt)
        lable_poxy_E = F.normalize(lable_poxy)


        loss_info_i=loss_info(img_E,text_E ,lable_poxy_E,corrected_labels,clean_mask)
        loss_info_t=loss_info(text_E ,img_E, lable_poxy_E, corrected_labels,clean_mask)

        loss_info_all = loss_info_i + loss_info_t


        F_I = F.normalize(img_embedding, dim=1)
        F_T = F.normalize(text_embedding, dim=1)

        BI_BI = F_I.mm(F_I.t())
        BT_BT = F_T.mm(F_T.t())
        BI_BT = F_I.mm(F_T.t())
        BT_BI = F_T.mm(F_I.t())

        loss_s_BI_BI ,loss_s_data_BI_BI = loss_w(BI_BI, S ,S_mask)
        loss_s_BT_BT,loss_s_data_BT_BT = loss_w(BT_BT, S ,S_mask)
        loss_s_BI_BT,loss_s_data_BI_BT = loss_w(BI_BT, S ,S_mask)
        loss_s_BT_BI,loss_s_data_BT_BI = loss_w(BT_BI, S ,S_mask)

        loss_s=loss_s_BI_BI+ loss_s_BT_BT+ loss_s_BI_BT+loss_s_BT_BI
        loss_s_data = loss_s_data_BI_BI


        B = torch.sign((img_embedding + text_embedding))
        loss_quant = (F.mse_loss(img_embedding, B) / img_embedding.shape[0] / self.config.hash_lens) + (
                    F.mse_loss(text_embedding, B) / text_embedding.shape[0] / self.config.hash_lens)

        # loss_cra = self.ContrastiveLoss(img_embedding, text_embedding)

        # loss =  loss_s+ loss_info_all+ loss_quant+loss_cra*0.1
        loss = loss_s + loss_info_all
        # loss =  loss_s+loss_cra_h+ loss_quant
        return loss,loss_s_data,loss_s



    def eval_retrieval(self,epoch):
        test_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
        retrieval_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
        self.ImageMlp.eval()
        self.TextMlp.eval()
        with torch.no_grad():
            for _, (data_I, data_T, data_L, index) in enumerate(self.query_loader):

                data_I, data_T = data_I.cuda(), data_T.cuda()
                label_t = data_L.cuda()

                img_query = self.ImageMlp(data_I)
                txt_query = self.TextMlp(data_T)

                Im_code = torch.sign(img_query)
                Txt_code = torch.sign(txt_query)

                test_dl_dict['img_code'].append(Im_code)
                test_dl_dict['txt_code'].append(Txt_code)
                test_dl_dict['label'].append(label_t)

            for _, (data_I, data_T, data_L,index) in enumerate(self.retrieval_loader):
                data_I, data_T = data_I.cuda(), data_T.cuda()
                label_t_db= data_L.cuda()

                img_retrieval = self.ImageMlp(data_I)
                txt_retrieval = self.TextMlp(data_T)

                Im_code = torch.sign(img_retrieval)
                Txt_code = torch.sign(txt_retrieval)

                retrieval_dl_dict['img_code'].append(Im_code)
                retrieval_dl_dict['txt_code'].append(Txt_code)
                retrieval_dl_dict['label'].append(label_t_db)

        query_img = torch.cat(test_dl_dict['img_code'], dim=0).cpu()
        query_txt = torch.cat(test_dl_dict['txt_code'], dim=0).cpu()
        query_label = torch.cat(test_dl_dict['label'], dim=0).cpu()

        retrieval_img = torch.cat(retrieval_dl_dict['img_code'], dim=0).cpu()
        retrieval_txt = torch.cat(retrieval_dl_dict['txt_code'], dim=0).cpu()
        retrieval_label = torch.cat(retrieval_dl_dict['label'], dim=0).cpu()
     
        mapi2t = calc_map_k(query_img.cuda(), retrieval_txt.cuda(), query_label.cuda(), retrieval_label.cuda())
        mapt2i = calc_map_k(query_txt.cuda(), retrieval_img.cuda(), query_label.cuda(), retrieval_label.cuda())

        if mapi2t + mapt2i > self.max_map['i2t'] + self.max_map['t2i']:
            self.max_map['i2t'] = mapi2t
            self.max_map['t2i'] = mapt2i
            self.best_epoch=epoch
            save_feature(
                         torch.as_tensor(self.train_loader.dataset.labs, dtype=torch.float32),
                         torch.as_tensor(self.train_loader.dataset.noise_label, dtype=torch.float32),
                         self.corrected_labels,
                         torch.as_tensor(self.train_loader.dataset.images, dtype=torch.float32),
                         torch.as_tensor(self.train_loader.dataset.texts, dtype=torch.float32),
                         self.config.dataset, noisy_num=0.8,epoch=None
                         )

            save_checkpoints(self)
            save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, query_label, retrieval_label)
        self.logger.info("best epoch : {}".format(self.best_epoch))
        self.logger.info("max_mAPi2t:{}, max_mAPt2i:{}".format(self.max_map['i2t'],self.max_map['t2i']))

        log_map_results(self,epoch, mapi2t, mapt2i)
        return mapi2t.item(), mapt2i.item()

    def train(self):
            self.max_map['i2t'] = self.max_map['t2i']
            I2T_MAP = []
            T2I_MAP = []

            starimt=time.time()
            for epoch in range(self.config.epoch):
                self.logger.info("=============== epoch: {}===============".format(epoch + 1))
                train_loss= self.train_epoch(epoch)

                torch.as_tensor(self.train_loader.dataset.noise_label, dtype=torch.float32),
                torch.as_tensor(self.train_loader.dataset.images, dtype=torch.float32),
                torch.as_tensor(self.train_loader.dataset.texts, dtype=torch.float32),

                self.logger.info("Training loss: {}".format(train_loss))
                end= time.time()
                print("epoch time: {}".format(end - starimt))
                if ((epoch + 1) % self.config.freq == 0) and (epoch + 1) > self.config.evl_epoch  :
                    self.logger.info("Testing...")
                    img2text, text2img = self.eval_retrieval(epoch)
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    self.logger.info('I2T: {}, T2I: {}'.format(img2text, text2img))

    def test(self):
        load_checkpoints(self)
        img2text, text2img = self.eval_retrieval()
        self.logger.info('I2T: {}, T2I: {}'.format(img2text, text2img))


class Proj_Pure_MLP(nn.Module):
    def __init__(self, in_features, out_features, middle_dim):
        super(Proj_Pure_MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.MLP = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, input):
        out = self.MLP(input)
        return out



def printlog(self,epoch,sort_ids):

    count = torch.isin(sort_ids, self.noise_index).sum().item()

    print("噪音的样的数量：", len(self.noise_index))
    print("选择为噪音的样的数量：", len(sort_ids))
    print("初步判断为噪音的样本实际为噪音的数量为：", count)
    print(f"初步判断为噪音的样本中，实际为噪音的比例: {count / (len(sort_ids)) * 100:.2f}%")
    print("已经排除在外的训练样本数量:", self.excluded_samples.sum().item())

    true_indices = self.excluded_samples.nonzero().squeeze()
    print("排除在外样本 实际为噪音的数量为：", torch.isin(self.noise_index, true_indices).sum().item())
    print("已修改的样本数量:", (self.sample_modification_count > 0).sum().item())

