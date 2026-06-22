
import os

import numpy as np
import torch
from matplotlib import pyplot as plt


def save_loss_realtime(self, step_loss, save_dir="logs/loss/"):
    """记录当前step的loss到日志文件"""
    # 确保日志目录存在
    os.makedirs(save_dir, exist_ok=True)

    file = "loss.txt"
    filename = f"{self.log_filename}_{file}"
    save_path = os.path.join(save_dir, filename)

    # 追加写入当前loss
    with open(save_path, 'a') as f:
        f.write(f"{self.step_counts},{step_loss}\n")



def save_percentage(self, epoch, percentage, save_dir="logs/choice/"):
    """记录当前step的loss到日志文件"""
    # 确保日志目录存在
    os.makedirs(save_dir, exist_ok=True)

    file = "percentage.txt"
    filename = f"{self.log_filename}_{file}"
    save_path = os.path.join(save_dir, filename)

    with open(save_path, 'a') as f:
        f.write(f"{epoch},{percentage}\n")


def print_lossData(epoch,data,):
    plt.figure(figsize=(15, 5))
    # plt.plot(data, color='steelblue', alpha=0.6, linewidth=1, label=f'Loss\n(Epoch {epoch})')
    plt.plot(data, color='steelblue', alpha=0.6, linewidth=1)
    mean_loss = data.mean().item()
    # plt.axhline(y=mean_loss, color='green', linestyle='-', linewidth=2, alpha=0.9, label='Mean Loss')
    plt.axhline(y=mean_loss, color='green', linestyle='-', linewidth=2, alpha=0.9)
    # plt.ylabel('Loss', fontsize=12)
    plt.ylim(-0.18, 0.18)  # 固定Y轴范围为0到2
    plt.xlim(0, 5000)  # 固定Y轴范围为0到2
    # plt.grid(linestyle='--', alpha=0.5)
    plt.xticks([])  # 隐藏X轴刻度
    plt.yticks([])  # 隐藏Y轴刻度
    plt.legend()
    plt.show()


def save_double_loss_realtime(self, index, bs, step_counts, lossData_history,lossData,save_dir="logs/double_loss/"):
    if len(lossData_history) == lossData_history.maxlen:
        noise_losses = []  # 初始化
        clean_losses = []  # 初始化
        all_losses=[]
        for i in range(bs):
            current_index = index[i]

            Flue=True
            # Flue = False
            if Flue:
                t_minus1 = lossData_history[-1][index[i]] - lossData_history[-2][index[i]]  # v_{t-1}
                current_speed = lossData[index[i]] - lossData_history[-1][index[i]]  # v_t
                self.config.α=0.8
                loss_flu = self.config.α * torch.abs(current_speed) + (1 - self.config.α) * torch.abs(t_minus1)
                # loss_flu+lossData[index[i]]
            else:
                loss_flu = lossData[index[i]]



            # 分类存储
            if current_index in self.noise_index:
                noise_losses.append(loss_flu)
            else:
                clean_losses.append(loss_flu)
            all_losses.append(loss_flu)

        noise_losses_mean = sum(noise_losses) / len(noise_losses)
        clean_losses_mean = sum(clean_losses) / len(clean_losses)
        # all_losses_mean = sum(all_losses) / len(all_losses)

        # print("噪音数据损失,波动", noise_losses_mean)
        # print("纯净数据损失,波动", clean_losses_mean)

        """记录当前step的三类损失到CSV文件"""
        os.makedirs(save_dir, exist_ok=True)
        file = "flue.txt"
        filename = f"{self.log_filename}_{file}"
        save_path = os.path.join(save_dir, filename)

        # 追加当前数据
        with open(save_path, 'a') as f:
            # f.write(f"{step_counts},{noise_losses_mean},{clean_losses_mean},{all_losses_mean}\n")
            f.write(f"{step_counts},{noise_losses_mean},{clean_losses_mean}\n")


def save_double_loss_epoch(self, epoch,all_loss,save_dir="logs/double_loss/epoch/"):
    noise_loss = all_loss[self.noise_index].mean()
    real_loss = all_loss[self.real_index].mean()
    # print("总损失,波动", all_loss)
    # print("总噪音数据损失,波动", noise_loss)
    # print("总纯净数据损失,波动", real_loss)

    """记录当前step的三类损失到CSV文件"""
    os.makedirs(save_dir, exist_ok=True)
    file = "flue_epoch.txt"
    filename = f"{self.log_filename}_{file}"
    save_path = os.path.join(save_dir, filename)

    # 追加当前数据
    with open(save_path, 'a') as f:
        f.write(f"{epoch},{noise_loss },{real_loss}\n")

def log_map_results(self, epoch, mapi2t, mapt2i):

        save_dir= "logs/map/"

        os.makedirs(save_dir, exist_ok=True)

        file = "map.txt"
        filename = f"{self.log_filename}_{file}"
        save_path = os.path.join(save_dir, filename)
        # 写入到 txt 文件（追加模式）
        with open(save_path , "a") as f:
            f.write(f"{epoch},{mapi2t.item()},{mapt2i.item()}\n")


def all_flue():
    pass