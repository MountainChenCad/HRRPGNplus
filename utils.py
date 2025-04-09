import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import os
import pandas as pd
from config import Config
import math
import time
import datetime
import json
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import io
from PIL import Image
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Define CVPR-quality color palette
COLORS = ['#0783D5', '#E52119', '#FD751F', '#0E2D88', '#78196D',
          '#C2C121', '#FC837E', '#00A6BC', '#025057', '#7E5505', '#77196C']

# Set global matplotlib parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12


# Generate static adjacency matrix
def generate_distance_matrix(N):
    """Generate a more sophisticated static adjacency matrix"""
    distance_matrix = torch.zeros(N, N, dtype=torch.float32)

    # Combine both distance and window-based connectivity
    for i in range(N):
        for j in range(N):
            # Distance-based connectivity
            dist_weight = 1.0 / (abs(i - j) + 1)

            # Window-based connectivity (strong local connections)
            window_weight = 1.0 if abs(i - j) <= 5 else 0.0

            # Combined weighting
            distance_matrix[i, j] = 0.7 * dist_weight + 0.3 * window_weight

    return distance_matrix


def prepare_static_adjacency(batch_size, feature_size, device):
    """准备批量的静态邻接矩阵"""
    # 处理极端情况
    if feature_size <= 1:
        print(f"警告: 特征大小 {feature_size} 太小，使用默认大小500")
        feature_size = 500

    static_adj = generate_distance_matrix(feature_size)
    static_adj = static_adj.to(device)
    static_adj = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
    return static_adj


def normalize_adjacency(adj):
    """标准化邻接矩阵 (D^(-1/2) * A * D^(-1/2))"""
    batch_size, N, _ = adj.size()

    # 计算度矩阵
    D = torch.sum(adj, dim=2)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

    # 扩展为对角矩阵
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)

    # 归一化邻接矩阵
    normalized_adj = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)

    return normalized_adj


def compute_metrics(true_labels, pred_labels, class_names=None):
    """计算性能评估指标"""
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )

    # 计算每个类别的准确率
    cm = confusion_matrix(true_labels, pred_labels)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    # 创建类别名称字典
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(class_acc))]

    # 为每个类别计算精确率、召回率和F1分数
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )

    class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(class_acc):
            class_metrics[class_name] = {
                'accuracy': class_acc[i] * 100,
                'precision': per_class_precision[i] * 100,
                'recall': per_class_recall[i] * 100,
                'f1': per_class_f1[i] * 100
            }

    metrics = {
        'accuracy': acc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'class_metrics': class_metrics,
        'confusion_matrix': cm
    }

    return metrics


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None, normalize=False, figsize=(10, 8)):
    """绘制混淆矩阵"""
    plt.figure(figsize=figsize, facecolor='white')

    # 是否归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create custom colormap (white to dark blue)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#0783D5'])

    # 创建热力图
    ax = plt.subplot(111)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    # Configure axes
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # Add labels and title
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=10)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_learning_curve(train_accs, val_accs, train_losses=None, val_losses=None,
                        title='Learning Curve', save_path=None, figsize=(12, 10)):
    """绘制学习曲线"""
    fig = plt.figure(figsize=figsize, facecolor='white')
    epochs = list(range(1, len(train_accs) + 1))

    # 设置双y轴图表
    if train_losses is not None and val_losses is not None:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # 上图：精度曲线
        ax1 = plt.subplot(gs[0])
        ax1.plot(epochs, train_accs, '-', color=COLORS[0], linewidth=2.5, marker='o',
                 markersize=5, label='Training Accuracy')
        ax1.plot(epochs, val_accs, '-', color=COLORS[1], linewidth=2.5, marker='s',
                 markersize=5, label='Validation Accuracy')

        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f"{title} - Accuracy", fontsize=14, fontweight='bold', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', which='major', width=1.5, length=5)

        # 下图：损失曲线
        ax2 = plt.subplot(gs[1])
        ax2.plot(epochs, train_losses, '-', color=COLORS[0], linewidth=2.5, marker='o',
                 markersize=5, label='Training Loss')
        ax2.plot(epochs, val_losses, '-', color=COLORS[1], linewidth=2.5, marker='s',
                 markersize=5, label='Validation Loss')

        ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title(f"{title} - Loss", fontsize=14, fontweight='bold', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', which='major', width=1.5, length=5)

        # Use integer ticks for epochs
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    else:
        # 只有精度曲线
        ax = plt.subplot(111)
        ax.plot(epochs, train_accs, '-', color=COLORS[0], linewidth=2.5, marker='o',
                markersize=5, label='Training Accuracy')
        ax.plot(epochs, val_accs, '-', color=COLORS[1], linewidth=2.5, marker='s',
                markersize=5, label='Validation Accuracy')

        ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=1.5, length=5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_shot_curve(shot_sizes, accuracies, ci=None, methods=None, f1_scores=None,
                    title='Shot-Accuracy Curve', save_path=None, figsize=(10, 6)):
    """绘制样本数-精度曲线"""
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = plt.subplot(111)

    if methods is None:
        # 单模型，有误差条带
        if ci is not None:
            ax.errorbar(shot_sizes, accuracies, yerr=ci, fmt='-o', capsize=5,
                        linewidth=2.5, markersize=8, color=COLORS[0],
                        ecolor=COLORS[0], elinewidth=1.5, label='Accuracy')
        else:
            ax.plot(shot_sizes, accuracies, '-o', linewidth=2.5, markersize=8,
                    color=COLORS[0], label='Accuracy')

        # 绘制F1分数曲线（如果提供）
        if f1_scores is not None:
            ax.plot(shot_sizes, f1_scores, '--s', linewidth=2.5, markersize=8,
                    color=COLORS[2], label='F1 Score')
    else:
        # 多模型对比
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        for i, (method_name, method_accs) in enumerate(zip(methods, accuracies)):
            color = COLORS[i % len(COLORS)]
            marker = markers[i % len(markers)]
            ax.plot(shot_sizes, method_accs, marker=marker, linewidth=2.5, markersize=8,
                    label=method_name, color=color)

            # Add data point labels
            for j, (x, y) in enumerate(zip(shot_sizes, method_accs)):
                if j == len(shot_sizes) - 1:  # Only label the last point to avoid clutter
                    ax.annotate(f'{y:.1f}%', (x, y), xytext=(5, 0),
                                textcoords='offset points', ha='left', fontsize=9)

    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Use integer ticks for shot sizes
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_features(features, labels, title='t-SNE Feature Visualization',
                       save_path=None, method='tsne', class_names=None, figsize=(12, 10)):
    """使用降维方法可视化特征"""
    # 标准化特征
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # 降维到2D
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=Config.seed)
        features_2d = reducer.fit_transform(features)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=Config.seed)
        features_2d = reducer.fit_transform(features)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=Config.seed)
        features_2d = reducer.fit_transform(features)
    else:
        raise ValueError(f"不支持的降维方法: {method}")

    # 绘制散点图
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = plt.subplot(111)

    unique_labels = np.unique(labels)

    # 创建类名映射
    if class_names is None:
        class_names = {i: f"Class {i}" for i in unique_labels}
    elif not isinstance(class_names, dict):
        class_names = {i: name for i, name in enumerate(class_names) if i in unique_labels}

    # 绘制每个类别的样本
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names.get(label, f"Class {label}")

        # Use the specified color palette
        color = COLORS[i % len(COLORS)]

        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=color, label=label_name, s=80, alpha=0.7,
            edgecolors='k', linewidths=1
        )

    # Styling
    ax.set_title(f"{title} ({method.upper()})", fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return features_2d


def visualize_dynamic_graph(adjacency, save_path=None, title='Dynamic Graph Adjacency Matrix',
                            threshold=None, figsize=(12, 10), show_network=True):
    """可视化动态图结构"""
    fig = plt.figure(figsize=figsize, facecolor='white')

    # 转换为numpy数组
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.detach().cpu().numpy()

    # 应用阈值（如果指定）
    if threshold is not None:
        adj_viz = adjacency.copy()
        adj_viz[adj_viz < threshold] = 0
    else:
        adj_viz = adjacency

    # 创建分割的绘图区域
    if show_network:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Left: Adjacency Matrix Heatmap
        # Custom colormap from white to blue
        cmap = LinearSegmentedColormap.from_list('custom_blue', ['#FFFFFF', '#0783D5'])
        im = ax1.imshow(adj_viz, cmap=cmap)
        ax1.set_title('Adjacency Matrix', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Node Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Node Index', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', which='major', width=1.5, length=5)

        # Add colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        # Right: Network Visualization
        # For visualization, select the first N nodes
        N = min(50, adj_viz.shape[0])  # Maximum 50 nodes for clarity
        sub_adj = adj_viz[:N, :N]

        # Create network graph
        G = nx.from_numpy_array(sub_adj)
        pos = nx.spring_layout(G, seed=Config.seed)

        # Calculate edge weights and node degrees
        edge_weights = [sub_adj[u, v] * 3 for u, v in G.edges()]
        node_degrees = [sum(sub_adj[i, :]) * 50 for i in range(N)]

        # Create a custom colormap for nodes
        node_cmap = plt.cm.get_cmap('viridis')
        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_degrees, alpha=0.8,
                                       node_color=node_degrees, cmap=node_cmap, ax=ax2)

        # Draw edges with alpha proportional to weight
        edges = nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6,
                                       edge_color='gray', ax=ax2)

        # Add node labels (only for a subset)
        labels = {i: str(i) for i in range(0, N, max(1, N // 10))}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax2)

        ax2.set_title(f'Network Visualization (First {N} Nodes)',
                      fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')

        # Add a colorbar for the node colors
        sm = plt.cm.ScalarMappable(cmap=node_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.1,
                            label='Node Degree')
        cbar.set_label('Node Degree', fontsize=12, fontweight='bold')

    else:
        # 只显示热力图
        ax = plt.subplot(111)
        # Custom colormap from white to blue
        cmap = LinearSegmentedColormap.from_list('custom_blue', ['#FFFFFF', '#0783D5'])
        im = ax.imshow(adj_viz, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Node Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Node Index', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', width=1.5, length=5)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_attention(data, attention_weights, title='Attention Weights',
                        save_path=None, figsize=(14, 8)):
    """可视化注意力权重"""
    fig = plt.figure(figsize=figsize, facecolor='white')

    # 确保数据是numpy数组
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy().flatten()
    else:
        data_np = np.array(data).flatten()

    # 确保注意力权重是numpy数组
    if isinstance(attention_weights, torch.Tensor):
        attn_np = attention_weights.detach().cpu().numpy().flatten()
    else:
        attn_np = np.array(attention_weights).flatten()

    # 上半部分：HRRP信号
    ax1 = plt.subplot(2, 1, 1)
    time_axis = np.arange(len(data_np))
    ax1.plot(time_axis, np.abs(data_np), color=COLORS[0], linewidth=2.5)
    ax1.set_title('HRRP Signal Magnitude', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 下半部分：注意力权重与信号叠加
    ax2 = plt.subplot(2, 1, 2)

    # 左侧Y轴：绘制信号
    ax2.plot(time_axis, np.abs(data_np), color=COLORS[0], linewidth=2.5, alpha=0.5,
             label='HRRP Signal')
    ax2.set_xlabel('Range Cell', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', width=1.5, length=5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 右侧y轴：注意力权重
    ax3 = ax2.twinx()

    # 根据注意力权重的维度调整
    if len(attn_np) == len(data_np):
        # 注意力和信号长度一致的情况
        ax3.bar(time_axis, attn_np, alpha=0.5, color=COLORS[1],
                label='Attention Weight', width=1.0)
    else:
        # 注意力维度不一致时进行插值
        interp_attn = np.interp(
            np.linspace(0, 1, len(data_np)),
            np.linspace(0, 1, len(attn_np)),
            attn_np
        )
        ax3.bar(time_axis, interp_attn, alpha=0.5, color=COLORS[1],
                label='Attention Weight', width=1.0)

    ax3.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=COLORS[1])
    ax3.spines['top'].set_visible(False)

    # 设置标题
    ax2.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 设置图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_attention_heatmap(data, attention_matrix, title='Attention Heatmap',
                                save_path=None, figsize=(15, 10)):
    """可视化注意力热力图"""
    fig = plt.figure(figsize=figsize, facecolor='white')

    # 确保数据是numpy数组
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy().flatten()
    else:
        data_np = np.array(data).flatten()

    # 确保注意力矩阵是numpy数组
    if isinstance(attention_matrix, torch.Tensor):
        attn_np = attention_matrix.detach().cpu().numpy()
    else:
        attn_np = np.array(attention_matrix)

    # 上半部分：HRRP信号
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.abs(data_np), color=COLORS[0], linewidth=2.5)
    ax1.set_title('HRRP Signal Magnitude', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 下半部分：注意力热力图
    ax2 = plt.subplot(2, 1, 2)

    # Custom colormap from white to blue
    cmap = LinearSegmentedColormap.from_list('custom_attention', ['#FFFFFF', '#0783D5'])
    im = ax2.imshow(attn_np, aspect='auto', cmap=cmap)

    ax2.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Attention Target', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attention Source', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', width=1.5, length=5)

    # Add colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def log_metrics(metrics, epoch=None, prefix='', file_path=None):
    """记录和打印性能指标"""
    log_str = f"{prefix} "
    if epoch is not None:
        log_str += f"Epoch {epoch}: "

    log_str += f"Accuracy: {metrics['accuracy']:.2f}%, "
    log_str += f"Precision: {metrics['precision']:.2f}%, "
    log_str += f"Recall: {metrics['recall']:.2f}%, "
    log_str += f"F1: {metrics['f1']:.2f}%"

    print(log_str)

    # 记录每个类别的详细指标
    if 'class_metrics' in metrics:
        for class_name, class_metric in metrics['class_metrics'].items():
            class_log = f"{prefix} {class_name}: "
            class_log += f"Acc: {class_metric['accuracy']:.2f}%, "
            class_log += f"Prec: {class_metric['precision']:.2f}%, "
            class_log += f"Rec: {class_metric['recall']:.2f}%, "
            class_log += f"F1: {class_metric['f1']:.2f}%"
            print(class_log)

    # 将日志写入文件
    if file_path:
        with open(file_path, 'a') as f:
            f.write(log_str + '\n')

            if 'class_metrics' in metrics:
                for class_name, class_metric in metrics['class_metrics'].items():
                    class_log = f"{prefix} {class_name}: "
                    class_log += f"Acc: {class_metric['accuracy']:.2f}%, "
                    class_log += f"Prec: {class_metric['precision']:.2f}%, "
                    class_log += f"Rec: {class_metric['recall']:.2f}%, "
                    class_log += f"F1: {class_metric['f1']:.2f}%"
                    f.write(class_log + '\n')

    return log_str


def save_model(model, save_path):
    """保存模型"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model, load_path):
    """加载模型"""
    if not os.path.exists(load_path):
        print(f"Warning: Model file {load_path} does not exist")
        return model

    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")
    return model


def set_requires_grad(model, requires_grad=True):
    """设置模型是否需要梯度"""
    for param in model.parameters():
        param.requires_grad = requires_grad


def create_experiment_log(config):
    """创建实验日志"""
    # 确保日志目录存在
    os.makedirs(config.log_dir, exist_ok=True)

    log_path = os.path.join(config.log_dir, 'experiment_config.txt')
    with open(log_path, 'w') as f:
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")

    # 创建JSON格式的配置文件，方便后续分析
    json_log_path = os.path.join(config.log_dir, 'experiment_config.json')
    with open(json_log_path, 'w') as f:
        # 创建可序列化的配置字典
        config_dict = {}
        for key, value in vars(config).items():
            if not key.startswith('__'):
                # 确保值是可序列化的
                if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                    config_dict[key] = value
                else:
                    # 尝试转换不可序列化的类型
                    try:
                        if isinstance(value, np.ndarray):
                            config_dict[key] = value.tolist()
                        elif isinstance(value, torch.Tensor):
                            config_dict[key] = value.cpu().numpy().tolist()
                        else:
                            config_dict[key] = str(value)
                    except:
                        config_dict[key] = str(value)

        json.dump(config_dict, f, indent=4)

    print(f"Experiment configuration logged to {log_path} and {json_log_path}")


# MAML++ Specific Utility Functions

def adapt_parameters(model, loss, lr=0.01, params=None):
    """Update parameters based on loss, return updated parameter dictionary"""
    if params is None:
        params = {n: p.clone() for n, p in model.named_parameters()}

    # Compute gradients
    grads = torch.autograd.grad(loss, params.values(), create_graph=True, allow_unused=True)

    # Update parameters
    updated_params = {}
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            updated_params[name] = param
        else:
            updated_params[name] = param - lr * grad

    return updated_params


def compute_cosine_annealing_lr(epoch, max_epochs, min_lr, max_lr):
    """Compute learning rate using cosine annealing schedule"""
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs))


def plot_learning_rates(layer_lrs, save_path=None):
    """Plot the learning rates for different layers across steps"""
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.subplot(111)

    for i, (layer_name, step_lrs) in enumerate(layer_lrs.items()):
        # Convert tensor values to numpy array
        if isinstance(step_lrs[0], torch.Tensor):
            step_lrs = [lr.item() for lr in step_lrs]

        # Use colors from the palette
        color = COLORS[i % len(COLORS)]

        ax.plot(range(len(step_lrs)), step_lrs, marker='o', linewidth=2.5,
                markersize=6, label=layer_name, color=color)

    # Styling
    ax.set_title("Per-Layer Learning Rates", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Step", fontsize=12, fontweight='bold')
    ax.set_ylabel("Learning Rate", fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def save_learning_rates(layer_lrs, file_path):
    """Save learning rates to CSV file"""
    df = pd.DataFrame(layer_lrs)
    df.to_csv(file_path, index_label='Step')
    print(f"Learning rates saved to {file_path}")


def create_params_dict_from_names(model):
    """Create parameter dictionary with name as key and parameter as value"""
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param
    return params_dict


def clone_params_dict(params_dict):
    """Deep clone a parameter dictionary"""
    return {name: param.clone() for name, param in params_dict.items()}


def log_gradient_norm(model, phase='train'):
    """Log the gradient norm for model parameters"""
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"{phase} gradient norm: {total_norm}")
    return total_norm


def find_latest_experiment():
    """Find the latest experiment ID"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return None

    experiment_dirs = [d for d in os.listdir(logs_dir) if d.startswith("experiment_")]
    if not experiment_dirs:
        return None

    # Sort by timestamp to get the latest
    latest_exp = sorted(experiment_dirs, reverse=True)[0]
    return latest_exp.replace("experiment_", "")


def get_module_by_name(model, module_name):
    """Get a module from a model by its name"""
    names = module_name.split('.')
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def analyze_model_complexity(model):
    """Analyze model complexity and parameter distribution"""
    # Count parameters by layer
    layer_params = {}
    total_params = 0

    for name, param in model.named_parameters():
        layer_name = name.split('.')[0:2]
        layer_name = '.'.join(layer_name)

        num_params = param.numel()
        total_params += num_params

        if layer_name in layer_params:
            layer_params[layer_name] += num_params
        else:
            layer_params[layer_name] = num_params

    # Print parameter distribution
    print(f"Model has {total_params} total parameters")
    print("\nParameter distribution by layer:")
    for layer_name, num_params in layer_params.items():
        percentage = (num_params / total_params) * 100
        print(f"{layer_name}: {num_params} parameters ({percentage:.2f}%)")

    return layer_params, total_params


def calculate_model_memory(model):
    """Calculate memory usage of the model in MB"""
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = (mem_params + mem_bufs) / 1024 / 1024  # convert to MB
    return mem_total


def timing_measurement(model, input_data, device, num_runs=100):
    """Measure model inference time"""
    model.eval()

    # Move model and input to the same device
    model = model.to(device)
    input_data = input_data.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(input_data)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)

    # Synchronize after timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate average time
    avg_time = (end_time - start_time) * 1000 / num_runs  # ms

    return avg_time


def create_comparison_table(results_dict, metrics=None, save_path=None):
    """Create and optionally save a comparison table from results"""
    if metrics is None:
        metrics = ['accuracy', 'f1', 'precision', 'recall']

    # Create DataFrame
    data = []
    for model_name, results in results_dict.items():
        row = {'model': model_name}
        for metric in metrics:
            if metric in results:
                row[metric] = results[metric]
        data.append(row)

    df = pd.DataFrame(data)

    # Print the table
    print("\nComparison Table:")
    print(df.to_string(index=False))

    # Save to CSV if path specified
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    return df


def fig_to_image(fig):
    """Convert a matplotlib figure to a PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def create_model_summary(model, input_shape=(1, 1, Config.feature_size), save_path=None):
    """Create a summary of the model architecture"""
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create a dictionary to store layer information
    layers_info = []

    # Collect layer information
    for name, module in model.named_modules():
        if name == '':  # Skip the top level module
            continue

        # Get layer type
        layer_type = module.__class__.__name__

        # Get parameters for this layer
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Add to info list
        layers_info.append({
            'name': name,
            'type': layer_type,
            'parameters': params
        })

    # Create DataFrame
    df = pd.DataFrame(layers_info)

    # Print summary
    print("\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Number of Layers: {len(layers_info)}")
    print("\nLayer Information:")
    print(df.to_string(index=False))

    # Save to file if path specified
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Model Summary:\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Number of Layers: {len(layers_info)}\n\n")
            f.write("Layer Information:\n")
            f.write(df.to_string(index=False))
        print(f"Model summary saved to {save_path}")

    return df


def visualize_model_parameters(model, save_path=None):
    """Visualize parameter distributions by layer"""
    # Collect parameter statistics by layer
    layer_stats = {}

    for name, param in model.named_parameters():
        # Extract base layer name
        layer_name = '.'.join(name.split('.')[:2])

        # Skip if parameter doesn't require grad
        if not param.requires_grad:
            continue

        # Get parameter data
        param_data = param.detach().cpu().numpy().flatten()

        # Calculate statistics
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                'count': 0,
                'mean': [],
                'std': [],
                'min': [],
                'max': [],
                'abs_mean': [],
                'data': []
            }

        layer_stats[layer_name]['count'] += param_data.size
        layer_stats[layer_name]['mean'].append(param_data.mean())
        layer_stats[layer_name]['std'].append(param_data.std())
        layer_stats[layer_name]['min'].append(param_data.min())
        layer_stats[layer_name]['max'].append(param_data.max())
        layer_stats[layer_name]['abs_mean'].append(np.abs(param_data).mean())

        # Store a sample of data for histogram
        if len(param_data) > 1000:
            layer_stats[layer_name]['data'].append(np.random.choice(param_data, 1000))
        else:
            layer_stats[layer_name]['data'].append(param_data)

    # Create visualization
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # 1. Parameter count by layer
    ax1 = plt.subplot(gs[0, 0])
    layers = list(layer_stats.keys())
    param_counts = [layer_stats[layer]['count'] for layer in layers]

    # Use colors from the palette
    colors = [COLORS[i % len(COLORS)] for i in range(len(layers))]

    ax1.bar(range(len(layers)), param_counts, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right', fontsize=10)
    ax1.set_title('Parameter Count by Layer', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)

    # 2. Parameter statistics by layer
    ax2 = plt.subplot(gs[0, 1])
    means = [np.mean(layer_stats[layer]['mean']) for layer in layers]
    stds = [np.mean(layer_stats[layer]['std']) for layer in layers]

    ax2.errorbar(range(len(layers)), means, yerr=stds, fmt='o', capsize=5,
                 linewidth=2, markersize=8, ecolor='gray')

    for i, (x, y) in enumerate(zip(range(len(layers)), means)):
        ax2.scatter([x], [y], s=100, color=colors[i])

    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha='right', fontsize=10)
    ax2.set_title('Parameter Mean and Std by Layer', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='major', width=1.5, length=5)

    # 3. Parameter distribution histograms
    ax3 = plt.subplot(gs[1, 0])
    for i, layer in enumerate(layers):
        data = np.concatenate(layer_stats[layer]['data'])
        ax3.hist(data, bins=50, alpha=0.6, label=layer, color=colors[i])

    ax3.set_title('Parameter Distribution', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='both', which='major', width=1.5, length=5)

    # 4. Absolute mean by layer
    ax4 = plt.subplot(gs[1, 1])
    abs_means = [np.mean(layer_stats[layer]['abs_mean']) for layer in layers]

    ax4.bar(range(len(layers)), abs_means, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(layers)))
    ax4.set_xticklabels(layers, rotation=45, ha='right', fontsize=10)
    ax4.set_title('Absolute Mean by Layer', fontsize=14, fontweight='bold', pad=10)
    ax4.set_ylabel('Absolute Mean', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.tick_params(axis='both', which='major', width=1.5, length=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return layer_stats


def create_experiment_report(experiment_id=None, config=None, results=None, save_dir=None):
    """Generate a comprehensive experiment report"""
    # Use latest experiment if none specified
    if experiment_id is None:
        experiment_id = find_latest_experiment()
        if experiment_id is None:
            print("No experiment found. Cannot generate report.")
            return

    # Set up paths
    if save_dir is None:
        save_dir = f"logs/experiment_{experiment_id}/report"

    os.makedirs(save_dir, exist_ok=True)

    # Load config if not provided
    if config is None:
        config_path = f"logs/experiment_{experiment_id}/experiment_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"Config file not found at {config_path}")
            config = {}

    # Generate report
    report_path = os.path.join(save_dir, 'experiment_report.md')

    with open(report_path, 'w') as f:
        # Header
        f.write(f"# Experiment Report: {experiment_id}\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")

        for key, value in config.items():
            # Skip complex nested structures
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                f.write(f"| {key} | (complex structure) |\n")
            else:
                f.write(f"| {key} | {value} |\n")

        # Results
        if results:
            f.write("\n## Results\n\n")

            # Overall metrics
            if 'metrics' in results:
                f.write("### Overall Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric, value in results['metrics'].items():
                    if not isinstance(value, dict) and not isinstance(value, np.ndarray):
                        f.write(f"| {metric} | {value:.2f}% |\n")

                # Class metrics
                if 'class_metrics' in results['metrics']:
                    f.write("\n### Class Metrics\n\n")
                    f.write("| Class | Accuracy | Precision | Recall | F1 |\n")
                    f.write("|-------|----------|-----------|--------|----|\n")

                    for class_name, metrics in results['metrics']['class_metrics'].items():
                        f.write(
                            f"| {class_name} | {metrics['accuracy']:.2f}% | {metrics['precision']:.2f}% | {metrics['recall']:.2f}% | {metrics['f1']:.2f}% |\n")

            # Learning curves
            if 'learning_curves' in results:
                f.write("\n### Learning Curves\n\n")
                f.write(f"![Learning Curves]({results['learning_curves']})\n")

            # Confusion matrix
            if 'confusion_matrix' in results:
                f.write("\n### Confusion Matrix\n\n")
                f.write(f"![Confusion Matrix]({results['confusion_matrix']})\n")

            # Additional visualizations
            if 'visualizations' in results:
                f.write("\n### Additional Visualizations\n\n")
                for name, path in results['visualizations'].items():
                    f.write(f"#### {name}\n\n")
                    f.write(f"![{name}]({path})\n\n")

        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This section should contain a brief summary of the experiment results and findings.\n")

    print(f"Experiment report generated at {report_path}")
    return report_path