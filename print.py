
import numpy as np
from scipy import stats

def printPltGmm(self, gmm, loss_data, threshold_prob, epoch):

    loss_data = loss_data.cpu().numpy()
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # 3. Create figure
    plt.figure(figsize=(10, 4))

    # Draw histogram
    plt.hist(loss_data,
             bins=100,
             density=True,
             alpha=0.6,
             color='gray',
             edgecolor='navy',
             linewidth=0.5,
             label='Loss Distribution')

    # 5. Draw GMM fitted curve
    x = np.linspace(np.min(loss_data), np.max(loss_data), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    plt.plot(x, pdf, 'k-', linewidth=2, label='GMM Fit')

    # 6. Draw two Gaussian components with shaded areas
    colors = ['#bde0b3', '#f3a86d']

    # plt.plot(steps, noise_losses, color='#f3a86d', linestyle='-', alpha=0.3, label='Raw Loss noise')
    # plt.plot(steps, smoothed_noise, color='#ee822f', linestyle='-', linewidth=2, label='Smoothed Loss noise')
    # plt.plot(steps, clean_losses, color='#bde0b3', linestyle='-', alpha=0.4, label='Raw Loss clean')
    # plt.plot(steps, smoothed_clean, color='#a0d292', linestyle='-', linewidth=2, label='Smoothed Loss clean')

    for k in range(2):
        pdf_component = (gmm.weights_[k] *
                         np.exp(-0.5 * ((x - gmm.means_[k]) / np.sqrt(gmm.covariances_[k][0])) ** 2) /
                         (np.sqrt(2 * np.pi) * np.sqrt(gmm.covariances_[k][0])))

        # Fill shaded area
        plt.fill_between(x.ravel(), pdf_component.ravel(),
                         color=colors[k],
                         alpha=0.5,
                         label=f'Component {k + 1} ')

        edge_colors = ['#a0d292', '#ee822f']  # Darker edge colors
        # Optional: Keep original curve (with thinner line)
        plt.plot(x, pdf_component,
                 linestyle='-',
                 color=edge_colors[k],
                 linewidth=2,
                 alpha=0.8)

    # Convert probability threshold to loss value threshold
    threshold_loss = norm.ppf(
        threshold_prob,
        loc=gmm.means_[1][0],
        scale=np.sqrt(gmm.covariances_[1][0])
    )

    plt.axvline(x=threshold_loss, color='red', linestyle=':',label=f'Noise Threshold')

    # 8. Add legend and labels
    plt.xlabel('Loss Value', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.xticks(fontsize=14)  # x轴刻度字号
    plt.yticks(fontsize=14)  # y轴刻度字号
    # plt.title('Loss Distribution with GMM Components and Noise Threshold', fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    # 9. Show plot
    plt.tight_layout()
    plt.show()

    # 10. 打印关键参数
    # 3. 修正打印语句（关键修改点）


    print("GMM Parameters:")
    # 正确提取协方差值（取[0][0][0]或使用item()）
    print(f"- Component 1 (Clean): μ={gmm.means_[0][0]:.4f}, "
          f"σ={np.sqrt(gmm.covariances_[0][0][0]):.4f}, "
          f"weight={gmm.weights_[0]:.2f}")

    print(f"- Component 2 (Noise): μ={gmm.means_[1][0]:.4f}, "
          f"σ={np.sqrt(gmm.covariances_[1][0][0]):.4f}, "
          f"weight={gmm.weights_[1]:.2f}")

    # 确保计算结果是标量
    kurtosis_value = float(stats.kurtosis(loss_data, fisher=False))  # 显式转换为Python float
    skewness_value = float(stats.skew(loss_data))  # 同样处理偏度
    print("=== 分布形态诊断 ===")
    print(f"峰度(Kurtosis): {kurtosis_value:.4f}")  # 现在可以安全格式化
    print(f"偏度(Skewness): {skewness_value:.4f}")

    # 双峰性判断标准
    if kurtosis_value > 3.5:
        print("→ 高峰度提示可能存在双峰分布")
    elif kurtosis_value < 2.5:
        print("→ 低峰度提示可能是单峰分布")
    else:
        print("→ 中等峰度，需结合其他指标判断")

    if abs(skewness_value) > 1:
        print(f"→ 明显偏态(|偏度|={abs(skewness_value):.2f}>1)")