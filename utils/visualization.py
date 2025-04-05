"""
可视化工具模块 - 提供数据和结果可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
import itertools


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', cmap=plt.cm.Blues,
                          normalize=False, save_path=None, figsize=(10, 8)):
    """
    绘制混淆矩阵

    参数:
    - cm: 混淆矩阵
    - class_names: 类别名称列表
    - title: 图表标题
    - cmap: 颜色映射
    - normalize: 是否归一化
    - save_path: 保存路径
    - figsize: 图表大小
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)

    # 绘制混淆矩阵热图
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar=True)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_learning_curves(history, metrics=None, save_path=None, figsize=(14, 6)):
    """
    绘制学习曲线

    参数:
    - history: 训练历史记录字典
    - metrics: 要绘制的指标列表 (默认为None，绘制所有指标)
    - save_path: 保存路径
    - figsize: 图表大小
    """
    if metrics is None:
        # 找出包含'loss'或'acc'的所有指标
        metrics = [key for key in history.keys() if 'loss' in key or 'acc' in key]

    num_plots = len(metrics)
    plt.figure(figsize=figsize)

    # 为每个指标创建子图
    for i, metric in enumerate(metrics):
        plt.subplot(1, num_plots, i + 1)

        # 获取指标数据
        if metric in history:
            data = history[metric]
            epochs = range(1, len(data) + 1)

            # 绘制曲线
            plt.plot(epochs, data, 'o-', label=metric)

            # 添加对应的验证指标(如果存在)
            val_metric = 'val_' + metric if 'val_' + metric in history else None
            if val_metric and val_metric in history:
                val_data = history[val_metric]
                if len(val_data) == len(epochs):
                    plt.plot(epochs, val_data, 's-', label=val_metric)

            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Epoch')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tsne(features, labels, title='t-SNE Visualization', save_path=None,
              figsize=(10, 8), random_state=42, perplexity=30):
    """
    使用t-SNE绘制特征可视化

    参数:
    - features: 特征矩阵 [N, D]
    - labels: 标签数组 [N]
    - title: 图表标题
    - save_path: 保存路径
    - figsize: 图表大小
    - random_state: 随机状态
    - perplexity: t-SNE困惑度参数
    """
    # 转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 确保特征是二维的
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)

    # 降至2维以便可视化
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, n_iter=300)
    features_2d = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=figsize)

    # 获取唯一类别
    unique_labels = np.unique(labels)

    # 为每个类别分配不同颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=[colors[i]], label=f'Class {label}',
                    alpha=0.7, edgecolors='k', linewidths=0.5)

    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_graph(adj_matrix, node_features=None, node_labels=None, title='Graph Visualization',
               save_path=None, figsize=(10, 8), layout='spring', node_size=300):
    """
    可视化图结构

    参数:
    - adj_matrix: 邻接矩阵 [N, N]
    - node_features: 节点特征 [N, D] (可选)
    - node_labels: 节点标签 [N] (可选)
    - title: 图表标题
    - save_path: 保存路径
    - figsize: 图表大小
    - layout: 图布局算法 ('spring', 'circular', 'random', 'shell')
    - node_size: 节点大小
    """
    # 转换为numpy数组
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()

    if node_features is not None and isinstance(node_features, torch.Tensor):
        node_features = node_features.detach().cpu().numpy()

    if node_labels is not None and isinstance(node_labels, torch.Tensor):
        node_labels = node_labels.detach().cpu().numpy()

    # 创建图
    G = nx.from_numpy_array(adj_matrix)

    # 设置节点属性
    if node_labels is not None:
        label_dict = {i: f"Node {i}\nClass {label}" for i, label in enumerate(node_labels)}
        nx.set_node_attributes(G, label_dict, 'label')

    # 计算节点位置
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)

    plt.figure(figsize=figsize)

    # 设置节点颜色
    if node_labels is not None:
        unique_labels = np.unique(node_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        node_colors = [colors[node_labels[n]] for n in G.nodes()]
    else:
        node_colors = 'skyblue'

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_size, alpha=0.7)

    # 绘制边
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)

    # 绘制标签
    if node_labels is not None:
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_weights(attention_weights, x_labels=None, y_labels=None,
                           title='Attention Weights', save_path=None, figsize=(10, 8)):
    """
    可视化注意力权重

    参数:
    - attention_weights: 注意力权重矩阵
    - x_labels: x轴标签
    - y_labels: y轴标签
    - title: 图表标题
    - save_path: 保存路径
    - figsize: 图表大小
    """
    # 转换为numpy数组
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    sns.heatmap(attention_weights, annot=False, cmap='viridis',
                xticklabels=x_labels, yticklabels=y_labels)

    plt.title(title)
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shot_comparison(shot_values, baseline_results, meta_results,
                         title='Performance vs. Number of Shots',
                         save_path=None, figsize=(10, 6), show_values=True):
    """
    绘制不同shot设置下的性能比较图

    参数:
    - shot_values: shot值列表
    - baseline_results: 基线模型不同shot下的结果
    - meta_results: 元学习模型不同shot下的结果
    - title: 图表标题
    - save_path: 保存路径
    - figsize: 图表大小
    - show_values: 是否显示数值标签
    """
    plt.figure(figsize=figsize)

    # 提取准确率
    baseline_acc = [baseline_results[k]['accuracy'] for k in shot_values]
    meta_acc = [meta_results[k]['accuracy'] for k in shot_values]

    # 绘制曲线
    plt.plot(shot_values, baseline_acc, 'o-', label='HRRPGraphNet', color='blue')
    plt.plot(shot_values, meta_acc, 's-', label='Meta-HRRPNet', color='red')

    # 添加数值标签
    if show_values:
        for i, (b_acc, m_acc) in enumerate(zip(baseline_acc, meta_acc)):
            plt.text(shot_values[i], b_acc - 3, f"{b_acc:.1f}%",
                     ha='center', va='top', color='blue', fontweight='bold')
            plt.text(shot_values[i], m_acc + 1.5, f"{m_acc:.1f}%",
                     ha='center', va='bottom', color='red', fontweight='bold')

    # 计算提升百分比
    improvements = [(m - b) / b * 100 for b, m in zip(baseline_acc, meta_acc)]

    # 添加提升标签
    for i, imp in enumerate(improvements):
        plt.annotate(f"+{imp:.1f}%",
                     xy=(shot_values[i], (baseline_acc[i] + meta_acc[i]) / 2),
                     xytext=(10, 0), textcoords='offset points',
                     fontsize=8, color='green', fontweight='bold')

    plt.xlabel('Number of Samples per Class (K-shot)')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # 设置x轴为整数值
    plt.xticks(shot_values)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ablation_results(ablation_results, group_names=None,
                          title='Ablation Study Results', save_path=None, figsize=(12, 6)):
    """
    绘制消融实验结果

    参数:
    - ablation_results: 消融实验结果字典
    - group_names: 组名列表
    - title: 图表标题
    - save_path: 保存路径
    - figsize: 图表大小
    """
    if group_names is None:
        group_names = list(ablation_results.keys())

    # 提取准确率和方差
    mean_accuracies = []
    std_accuracies = []

    for group in group_names:
        if group in ablation_results:
            # 如果每个组包含多次运行结果
            if isinstance(ablation_results[group], list):
                accs = [r['metrics']['accuracy'] for r in ablation_results[group]]
                mean_accuracies.append(np.mean(accs))
                std_accuracies.append(np.std(accs))
            else:
                # 单次运行结果
                mean_accuracies.append(ablation_results[group]['metrics']['accuracy'])
                std_accuracies.append(0)

    plt.figure(figsize=figsize)

    # 绘制柱状图
    x = np.arange(len(group_names))
    bars = plt.bar(x, mean_accuracies, yerr=std_accuracies, capsize=6,
                   alpha=0.7, error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1))

    # 添加基准线
    if len(mean_accuracies) > 0:
        baseline_acc = mean_accuracies[0]
        plt.axhline(y=baseline_acc, color='r', linestyle='--', alpha=0.5,
                    label=f'Baseline: {baseline_acc:.1f}%')

    # 添加数值标签
    for i, v in enumerate(mean_accuracies):
        plt.text(i, v + std_accuracies[i] + 0.5, f"{v:.1f}%",
                 ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Model Variant')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks(x, group_names)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_noise_robustness(noise_results, noise_types, snr_levels,
                          title='Noise Robustness Comparison', save_path=None, figsize=(12, 8)):
    """
    绘制噪声鲁棒性对比图

    参数:
    - noise_results: 噪声实验结果字典
    - noise_types: 噪声类型列表
    - snr_levels: 信噪比水平列表
    - title: 图表标题
    - save_path: 保存路径
    - figsize: 图表大小
    """
    plt.figure(figsize=figsize)

    # 每种噪声类型一个子图
    for i, noise_type in enumerate(noise_types):
        plt.subplot(len(noise_types), 1, i + 1)

        # 提取性能数据
        standard_acc = [noise_results['standard'][noise_type][snr]['accuracy']
                        for snr in snr_levels]
        meta_acc = [noise_results['meta'][noise_type][snr]['accuracy']
                    for snr in snr_levels]

        # 绘制性能曲线
        plt.plot(snr_levels, standard_acc, 'o-', label='HRRPGraphNet')
        plt.plot(snr_levels, meta_acc, 's-', label='Meta-HRRPNet')

        # 添加干净数据性能参考线
        plt.axhline(y=noise_results['clean']['standard'], color='blue',
                    linestyle='--', alpha=0.5,
                    label=f'HRRPGraphNet (clean): {noise_results["clean"]["standard"]:.1f}%')
        plt.axhline(y=noise_results['clean']['meta'], color='red',
                    linestyle='--', alpha=0.5,
                    label=f'Meta-HRRPNet (clean): {noise_results["clean"]["meta"]:.1f}%')

        plt.xlabel('SNR (dB)' if i == len(noise_types) - 1 else '')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{noise_type.capitalize()} Noise')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()