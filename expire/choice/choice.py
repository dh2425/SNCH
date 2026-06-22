import matplotlib.pyplot as plt

file = r"D:\Users\24226\Desktop\papper\5\正式版2\papperFive-8\logs\choice\percentage_2025-08-01_18-11-14.txt"

# 读取txt文件
with open(file, 'r') as f:
    lines = f.readlines()

# 解析数据（跳过标题行）
epochs = []
percentages = []
for line in lines[1:]:  # 跳过第一行标题
    epoch, percentage = line.strip().split(',')
    epochs.append(int(epoch))
    percentages.append(float(percentage))

# 创建图表
plt.figure(figsize=(12, 7), dpi=100)  # 增大画布尺寸和DPI

# 绘制更平滑的曲线：
# 1. 使用更粗的线条宽度
# 2. 使用平滑的线型
# 3. 减少标记点密度（每5个点显示一个标记）
# 4. 开启抗锯齿
plt.plot(epochs, percentages,
         marker='o',
         markevery=5,  # 每5个点显示一个标记
         linestyle='-',
         linewidth=2.5,  # 更粗的线条
         color='royalblue',  # 更柔和的颜色
         alpha=0.8,  # 轻微透明
         antialiased=True)  # 开启抗锯齿

# 添加标题和标签
plt.title('Model Performance by Epoch', fontsize=14, pad=20)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # 更淡的网格线

# 设置坐标轴范围
plt.xlim(0, max(epochs)+1)
plt.ylim(min(percentages)-2, max(percentages)+2)  # 减少边距

# 突出显示最高点
max_idx = percentages.index(max(percentages))
plt.annotate(f'Max: {max(percentages):.2f}%',
             xy=(epochs[max_idx], percentages[max_idx]),
             xytext=(epochs[max_idx], percentages[max_idx]+1.5),
             ha='center',
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10)

# 调整布局
plt.tight_layout()

# 保存和显示图表
plt.savefig('smoothed_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()