import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import pandas as pd
from typing import Tuple, Dict, Any, List
import warnings

warnings.filterwarnings('ignore')

from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import torch

def load_dataset():
    '''
        load datasets : flickr25k, mscoco, nus-wide
    '''



    retrieval_loc = r"D:\Users\24226\Desktop\papper\5\papper5\papperFive-8-6\logs\CAAfeature\featureCAA_nus-wide_0.5.pkl"
    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = torch.tensor(data['orl_label'], dtype=torch.int64)
        noise_label = torch.tensor(data['noise_label'], dtype=torch.int64)
        corrected_labels= torch.tensor(data['corrected_labels'], dtype=torch.int64)
        train_images = torch.tensor(data['image'], dtype=torch.float32)
        train_texts = torch.tensor(data['text'], dtype=torch.float32)

    return  train_labels,noise_label,corrected_labels,train_images,train_texts

class MultiLabelCorrelationAnalyzer:
    """
    多标签相关性分析器 - 验证四种标签情况
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        np.random.seed(random_state)

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """预处理特征"""
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        return self.scaler.fit_transform(features)

    def generate_random_labels(self, n_samples: int, n_classes: int) -> np.ndarray:
        """
        生成完全不相关的随机标签
        """
        return np.random.randint(0, 2, (n_samples, n_classes))

    def generate_anti_labels(self, clean_labels: np.ndarray) -> np.ndarray:
        """
        生成与干净标签完全相反的反标签
        这样可以确保反标签与特征完全不相关
        """
        return 1 - clean_labels

    def generate_permuted_labels(self, clean_labels: np.ndarray) -> np.ndarray:
        """
        对每个标签列独立洗牌，确保与特征完全无关
        """
        n_samples, n_classes = clean_labels.shape
        permuted_labels = clean_labels.copy()

        for j in range(n_classes):
            np.random.shuffle(permuted_labels[:, j])

        return permuted_labels

    def generate_near_zero_correlation_labels(self, clean_labels: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """
        生成相关性接近0的标签
        方法：对每个标签列独立洗牌 + 添加随机噪声
        """
        n_samples, n_classes = clean_labels.shape
        uncorrelated_labels = clean_labels.copy()

        # 对每个标签列独立洗牌，彻底破坏与特征的关系
        for j in range(n_classes):
            np.random.shuffle(uncorrelated_labels[:, j])

        # 添加一些随机噪声进一步降低任何潜在的相关性
        noise = np.random.normal(0, 0.1, uncorrelated_labels.shape)
        uncorrelated_labels = np.clip(uncorrelated_labels + noise, 0, 1)

        return uncorrelated_labels

    def compute_cca_correlation(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """
        计算CCA相关性指标
        """
        results = {}

        try:
            if len(X) != len(Y):
                raise ValueError("X和Y的长度必须相同")

            # 图像特征 vs 标签
            cca_img = CCA(n_components=self.n_components)
            img_cca, label_cca_img = cca_img.fit_transform(X, Y)
            img_corrs = [abs(pearsonr(img_cca[:, i], label_cca_img[:, i])[0])
                         for i in range(self.n_components)]
            results['image_cca_mean'] = np.mean(img_corrs)
            results['image_cca_std'] = np.std(img_corrs)
            results['image_cca_max'] = np.max(img_corrs)

            # 文本特征 vs 标签
            cca_txt = CCA(n_components=self.n_components)
            txt_cca, label_cca_txt = cca_txt.fit_transform(X, Y)
            txt_corrs = [abs(pearsonr(txt_cca[:, i], label_cca_txt[:, i])[0])
                         for i in range(self.n_components)]
            results['text_cca_mean'] = np.mean(txt_corrs)
            results['text_cca_std'] = np.std(txt_corrs)
            results['text_cca_max'] = np.max(txt_corrs)

            # 平均CCA相关性
            results['overall_cca_mean'] = (results['image_cca_mean'] + results['text_cca_mean']) / 2

        except Exception as e:
            print(f"CCA计算错误: {e}")
            results['image_cca_mean'] = 0.0
            results['text_cca_mean'] = 0.0
            results['overall_cca_mean'] = 0.0
            results['image_cca_std'] = 0.0
            results['text_cca_std'] = 0.0
            results['image_cca_max'] = 0.0
            results['text_cca_max'] = 0.0

        return results

    def compute_mutual_information_multi_label(self, features: np.ndarray,
                                               labels: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """
        计算特征和多标签之间的互信息
        """
        results = {}

        try:
            # 图像特征互信息
            mi_scores_img = []
            for feature_idx in range(min(30, features.shape[1])):
                for label_idx in range(labels.shape[1]):
                    x_binned = np.digitize(features[:, feature_idx],
                                           bins=np.linspace(features[:, feature_idx].min(),
                                                            features[:, feature_idx].max(), n_bins))
                    y_binned = np.digitize(labels[:, label_idx],
                                           bins=np.linspace(labels[:, label_idx].min(),
                                                            labels[:, label_idx].max(), n_bins))
                    mi = mutual_info_score(x_binned, y_binned)
                    mi_scores_img.append(mi)

            results['image_mi_mean'] = np.mean(mi_scores_img) if mi_scores_img else 0.0
            results['image_mi_std'] = np.std(mi_scores_img) if mi_scores_img else 0.0

            # 文本特征互信息 (使用相同的特征处理)
            mi_scores_txt = []
            for feature_idx in range(min(30, features.shape[1])):
                for label_idx in range(labels.shape[1]):
                    x_binned = np.digitize(features[:, feature_idx],
                                           bins=np.linspace(features[:, feature_idx].min(),
                                                            features[:, feature_idx].max(), n_bins))
                    y_binned = np.digitize(labels[:, label_idx],
                                           bins=np.linspace(labels[:, label_idx].min(),
                                                            labels[:, label_idx].max(), n_bins))
                    mi = mutual_info_score(x_binned, y_binned)
                    mi_scores_txt.append(mi)

            results['text_mi_mean'] = np.mean(mi_scores_txt) if mi_scores_txt else 0.0
            results['text_mi_std'] = np.std(mi_scores_txt) if mi_scores_txt else 0.0

            # 平均互信息
            results['overall_mi_mean'] = (results['image_mi_mean'] + results['text_mi_mean']) / 2

        except Exception as e:
            print(f"互信息计算错误: {e}")
            results['image_mi_mean'] = 0.0
            results['text_mi_mean'] = 0.0
            results['overall_mi_mean'] = 0.0
            results['image_mi_std'] = 0.0
            results['text_mi_std'] = 0.0

        return results

    def compute_label_consistency(self, labels: np.ndarray,
                                  clean_labels: np.ndarray = None) -> Dict[str, float]:
        """
        计算标签一致性指标
        """
        metrics = {}

        try:
            # 标签稀疏性
            metrics['sparsity'] = 1 - np.mean(labels)

            # 标签多样性 (平均每个样本的标签数)
            if len(labels.shape) > 1:
                metrics['label_diversity'] = np.mean(np.sum(labels, axis=1))
            else:
                metrics['label_diversity'] = np.mean(labels)

            # 如果提供了干净标签，计算准确率
            if clean_labels is not None:
                if labels.shape == clean_labels.shape:
                    # 对于软标签，使用阈值0.5进行二值化
                    if np.max(labels) <= 1.0 and np.min(labels) >= 0.0:
                        pred_hard = (labels > 0.5).astype(int)
                        accuracy = np.mean(pred_hard == clean_labels)
                        metrics['accuracy_vs_clean'] = accuracy

            # 标签置信度统计
            if np.max(labels) <= 1.0 and np.min(labels) >= 0.0:
                metrics['mean_confidence'] = np.mean(labels)
                metrics['confidence_std'] = np.std(labels)
                # 不确定性 (熵)
                epsilon = 1e-8
                entropy = -np.sum(labels * np.log(labels + epsilon) +
                                  (1 - labels) * np.log(1 - labels + epsilon), axis=1)
                metrics['mean_entropy'] = np.mean(entropy)

        except Exception as e:
            print(f"标签一致性计算错误: {e}")

        return metrics

    def analyze_all_label_types(self, image_features: np.ndarray,
                                text_features: np.ndarray,
                                clean_labels: np.ndarray,
                                noisy_labels: np.ndarray,
                                corrected_labels: np.ndarray) -> Dict[str, Any]:
        """
        分析四种标签类型的相关性
        """
        print("=== 多标签相关性分析 ===")
        print(f"数据形状: 图像特征 {image_features.shape}, 文本特征 {text_features.shape}")
        print(f"标签形状: 干净 {clean_labels.shape}, 噪声 {noisy_labels.shape}, 修正 {corrected_labels.shape}")

        # 预处理特征
        image_processed = self.preprocess_features(image_features)
        text_processed = self.preprocess_features(text_features)

        # 生成完全不相关的随机标签
        n_samples, n_classes = clean_labels.shape
        random_labels = self.generate_random_labels(n_samples, n_classes)
        # random_labels =self.generate_anti_labels(clean_labels)
        # random_labels =self.generate_permuted_labels(clean_labels)
        # random_labels =self.generate_near_zero_correlation_labels(clean_labels,image_processed)
        results = {}

        # 1. 分析完全不相关标签
        print("\n1. 分析完全不相关标签...")
        results['random_labels'] = self.analyze_single_label_type(
            image_processed, text_processed, random_labels, "随机标签")

        # 2. 分析干净标签
        print("\n2. 分析干净标签...")
        results['clean_labels'] = self.analyze_single_label_type(
            image_processed, text_processed, clean_labels, "干净标签", clean_labels)

        # 3. 分析噪声标签
        print("\n3. 分析噪声标签...")
        results['noisy_labels'] = self.analyze_single_label_type(
            image_processed, text_processed, noisy_labels, "噪声标签", clean_labels)

        # 4. 分析修正标签
        print("\n4. 分析修正标签...")
        results['corrected_labels'] = self.analyze_single_label_type(
            image_processed, text_processed, corrected_labels, "修正标签", clean_labels)

        # 综合比较
        self.print_comprehensive_comparison(results)

        return results

    def analyze_single_label_type(self, image_features: np.ndarray,
                                  text_features: np.ndarray,
                                  labels: np.ndarray,
                                  label_name: str,
                                  clean_labels: np.ndarray = None) -> Dict[str, Any]:
        """
        分析单一标签类型的相关性
        """
        result = {}

        # CCA相关性分析
        cca_image = self.compute_cca_correlation(image_features, labels)
        cca_text = self.compute_cca_correlation(text_features, labels)

        result['cca'] = {
            'image': cca_image,
            'text': cca_text
        }

        # 互信息分析
        mi_image = self.compute_mutual_information_multi_label(image_features, labels)
        mi_text = self.compute_mutual_information_multi_label(text_features, labels)

        result['mutual_info'] = {
            'image': mi_image,
            'text': mi_text
        }

        # 标签一致性分析
        result['consistency'] = self.compute_label_consistency(labels, clean_labels)

        print(f"  {label_name}分析完成:")
        print(f"    CCA相关性: 图像 {cca_image['overall_cca_mean']:.4f}, 文本 {cca_text['overall_cca_mean']:.4f}")
        print(f"    互信息: 图像 {mi_image['overall_mi_mean']:.4f}, 文本 {mi_text['overall_mi_mean']:.4f}")

        return result

    def print_comprehensive_comparison(self, results: Dict[str, Any]):
        """打印综合比较结果"""
        print("\n" + "=" * 80)
        print("四种标签类型相关性综合比较")
        print("=" * 80)

        # 创建比较表格
        comparison_data = []

        for label_type in ['random_labels', 'noisy_labels', 'corrected_labels', 'clean_labels']:
            if label_type in results:
                data = results[label_type]

                row = {
                    '标签类型': label_type.replace('_', ' ').title(),
                    '图像CCA': data['cca']['image']['overall_cca_mean'],
                    '文本CCA': data['cca']['text']['overall_cca_mean'],
                    '平均CCA': (data['cca']['image']['overall_cca_mean'] +
                                data['cca']['text']['overall_cca_mean']) / 2,
                    '图像MI': data['mutual_info']['image']['overall_mi_mean'],
                    '文本MI': data['mutual_info']['text']['overall_mi_mean'],
                    '平均MI': (data['mutual_info']['image']['overall_mi_mean'] +
                               data['mutual_info']['text']['overall_mi_mean']) / 2,
                    '稀疏性': data['consistency'].get('sparsity', 0),
                    '标签多样性': data['consistency'].get('label_diversity', 0),
                    '准确率': data['consistency'].get('accuracy_vs_clean', 0)
                }
                comparison_data.append(row)

        # 创建DataFrame并打印
        df = pd.DataFrame(comparison_data)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print("\n相关性指标比较:")
        print(df.to_string(index=False))

        # 分析趋势
        print("\n" + "=" * 50)
        print("趋势分析")
        print("=" * 50)

        # 计算改进程度
        if 'random_labels' in results and 'noisy_labels' in results:
            random_cca = (results['random_labels']['cca']['image']['overall_cca_mean'] +
                          results['random_labels']['cca']['text']['overall_cca_mean']) / 2
            noisy_cca = (results['noisy_labels']['cca']['image']['overall_cca_mean'] +
                         results['noisy_labels']['cca']['text']['overall_cca_mean']) / 2
            corrected_cca = (results['corrected_labels']['cca']['image']['overall_cca_mean'] +
                             results['corrected_labels']['cca']['text']['overall_cca_mean']) / 2
            clean_cca = (results['clean_labels']['cca']['image']['overall_cca_mean'] +
                         results['clean_labels']['cca']['text']['overall_cca_mean']) / 2

            print(f"CCA相关性趋势:")
            print(f"  随机标签 → 噪声标签: {noisy_cca - random_cca:+.4f}")
            print(f"  噪声标签 → 修正标签: {corrected_cca - noisy_cca:+.4f}")
            print(f"  修正标签 → 干净标签: {clean_cca - corrected_cca:+.4f}")
            print(f"  总体改进: {clean_cca - random_cca:+.4f}")

        # 验证假设
        print("\n" + "=" * 50)
        print("假设验证")
        print("=" * 50)

        if (results['corrected_labels']['cca']['image']['overall_cca_mean'] >
                results['noisy_labels']['cca']['image']['overall_cca_mean'] and
                results['corrected_labels']['cca']['text']['overall_cca_mean'] >
                results['noisy_labels']['cca']['text']['overall_cca_mean']):
            print("✓ 假设验证通过: 修正标签比噪声标签与特征更相关")
        else:
            print("✗ 假设验证失败: 修正标签的相关性改进不明显")

def check_data_issues( image_features: np.ndarray, text_features: np.ndarray, labels: np.ndarray):
    """检查数据问题"""
    print("\n=== 数据检查 ===")

    # 检查特征尺度
    print(f"图像特征范围: [{image_features.min():.4f}, {image_features.max():.4f}]")
    print(f"文本特征范围: [{text_features.min():.4f}, {text_features.max():.4f}]")

    # 检查特征方差
    img_var = np.var(image_features, axis=0)
    txt_var = np.var(text_features, axis=0)
    print(f"图像特征方差: 平均{np.mean(img_var):.4f}, 最小{np.min(img_var):.4f}, 最大{np.max(img_var):.4f}")
    print(f"文本特征方差: 平均{np.mean(txt_var):.4f}, 最小{np.min(txt_var):.4f}, 最大{np.max(txt_var):.4f}")

    # 检查是否有常数特征
    zero_var_features_img = np.sum(img_var < 1e-8)
    zero_var_features_txt = np.sum(txt_var < 1e-8)
    print(f"零方差特征: 图像{zero_var_features_img}, 文本{zero_var_features_txt}")

# 使用示例
def comprehensive_analysis_demo():
    """综合分析演示"""

    # 创建分析器
    analyzer = MultiLabelCorrelationAnalyzer(n_components=2, random_state=42)
    train_labels, noise_labels, corrected_labels, train_images, train_texts = load_dataset()

    orl_labels = train_labels.cpu().numpy()[:5000]
    noisy_labels = noise_labels.cpu().numpy()[:5000]
    corrected_labels= corrected_labels.cpu().numpy()[:5000]
    images=train_images.cpu().numpy()[:5000]
    texts = train_texts.cpu().numpy()[:5000]
    check_data_issues(images, texts, orl_labels)
    # 执行分析
    results = analyzer.analyze_all_label_types(
        image_features=images,
        text_features=texts,
        clean_labels=orl_labels ,
        noisy_labels=noisy_labels,
        corrected_labels=corrected_labels
    )

    return results


if __name__ == "__main__":
    # 运行演示
    analysis_results = comprehensive_analysis_demo()