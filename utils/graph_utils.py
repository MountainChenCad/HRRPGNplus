"""
图操作工具模块 - 提供图处理和分析功能
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_distance_matrix(N, alpha=1.0, metric='inverse_distance'):
    """
    生成预定义的距离矩阵

    参数:
    - N: 矩阵大小 (N×N)
    - alpha: 距离衰减系数
    - metric: 距离度量方式 ('inverse_distance', 'exp_decay', 'gaussian')

    返回:
    - 距离矩阵 [N, N]
    """
    distance_matrix = torch.zeros(N, N, dtype=torch.float32)

    for i in range(N):
        for j in range(N):
            d = abs(i - j)  # 欧式距离

            if metric == 'inverse_distance':
                # 反比例衰减: 1 / (alpha*d + 1)
                distance_matrix[i, j] = 1.0 / (alpha * d + 1.0)
            elif metric == 'exp_decay':
                # 指数衰减: exp(-alpha*d)
                distance_matrix[i, j] = torch.exp(-alpha * d)
            elif metric == 'gaussian':
                # 高斯核: exp(-alpha*d^2)
                distance_matrix[i, j] = torch.exp(-alpha * d * d)
            else:
                raise ValueError(f"不支持的距离度量方式: {metric}")

    return distance_matrix


def construct_knn_graph(X, k=5, metric='euclidean'):
    """
    构建k近邻图

    参数:
    - X: 特征矩阵 [N, D]
    - k: 近邻数量
    - metric: 距离度量方式 ('euclidean', 'cosine')

    返回:
    - 邻接矩阵 [N, N]
    """
    N = X.shape[0]
    adj_matrix = torch.zeros(N, N, dtype=X.dtype, device=X.device)

    if metric == 'euclidean':
        # 计算欧式距离矩阵
        dist_matrix = torch.cdist(X, X, p=2)
    elif metric == 'cosine':
        # 计算余弦相似度
        X_norm = F.normalize(X, p=2, dim=1)
        sim_matrix = torch.mm(X_norm, X_norm.t())
        dist_matrix = 1 - sim_matrix
    else:
        raise ValueError(f"不支持的距离度量方式: {metric}")

    # 对于每个节点，找到k个最近的邻居
    _, indices = torch.topk(dist_matrix, k=k + 1, dim=1, largest=False)

    # 构建邻接矩阵 (不包括自身)
    for i in range(N):
        for j in indices[i, 1:]:  # 跳过第一个(自身)
            adj_matrix[i, j] = 1.0

    return adj_matrix


def compute_graph_statistics(adj_matrix):
    """
    计算图的统计特性

    参数:
    - adj_matrix: 邻接矩阵 [N, N]

    返回:
    - 图统计特性字典
    """
    # 确保是numpy数组
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    # 转换为NetworkX图
    G = nx.from_numpy_array(adj_matrix)

    # 计算图的基本统计量
    statistics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': np.mean([d for _, d in G.degree()]),
        'max_degree': max([d for _, d in G.degree()]),
        'min_degree': min([d for _, d in G.degree()])
    }

    # 计算聚类系数
    try:
        statistics['clustering_coef'] = nx.average_clustering(G)
    except:
        statistics['clustering_coef'] = np.nan

    # 计算度分布
    degree_hist = nx.degree_histogram(G)
    statistics['degree_distribution'] = degree_hist

    return statistics


def symmetrize_adj_matrix(adj_matrix):
    """
    使邻接矩阵对称化

    参数:
    - adj_matrix: 邻接矩阵 [N, N]

    返回:
    - 对称化的邻接矩阵
    """
    # 如果是PyTorch张量
    if isinstance(adj_matrix, torch.Tensor):
        return 0.5 * (adj_matrix + adj_matrix.transpose(0, 1))
    # 如果是NumPy数组
    elif isinstance(adj_matrix, np.ndarray):
        return 0.5 * (adj_matrix + adj_matrix.T)
    else:
        raise TypeError("邻接矩阵必须是PyTorch张量或NumPy数组")


def normalize_adj_matrix(adj_matrix):
    """
    对邻接矩阵进行归一化(用于GCN)

    参数:
    - adj_matrix: 邻接矩阵 [N, N]

    返回:
    - 归一化后的邻接矩阵
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)

    # 添加自环
    N = adj_matrix.shape[0]
    adj_matrix = adj_matrix + torch.eye(N, device=adj_matrix.device)

    # 计算度矩阵
    rowsum = adj_matrix.sum(dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    # 归一化: D^(-1/2) A D^(-1/2)
    normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_matrix), d_mat_inv_sqrt)

    return normalized_adj


def calculate_edge_importance(adj_matrix, node_features=None):
    """
    计算边的重要性

    参数:
    - adj_matrix: 邻接矩阵 [N, N]
    - node_features: 节点特征 [N, D] (可选)

    返回:
    - 边重要性矩阵 [N, N]
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)

    N = adj_matrix.shape[0]
    importance = torch.zeros_like(adj_matrix)

    # 如果提供了节点特征，则计算特征相似度
    if node_features is not None:
        # 转换为张量
        if isinstance(node_features, np.ndarray):
            node_features = torch.FloatTensor(node_features)

        # 计算余弦相似度
        features_norm = node_features / (node_features.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(features_norm, features_norm.t())

        # 基于特征相似度和连接强度计算重要性
        importance = adj_matrix * similarity
    else:
        # 仅使用连接强度作为重要性
        importance = adj_matrix

    return importance