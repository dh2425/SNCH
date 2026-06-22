import re

import matplotlib.pyplot as plt
import numpy as np



flie=r"D:\Users\24226\Desktop\papper\5\正式版2\papperFive-8-2\logs\lossData\08-09_10-25_lossData.txt"




with open(flie, 'r') as f:
    data_str = f.read().split()  # 按空格/换行分割字符串
    data = [float(x) for x in data_str]  # 转换为浮点数

print(data)
plt.figure(figsize=(15, 5))
plt.plot(data, color='steelblue', alpha=0.6, linewidth=1, label='Loss值')
plt.title('训练损失（Loss）波动趋势', fontsize=15)
plt.xlabel('迭代步数', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.show()