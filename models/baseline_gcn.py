"""
原始HRRPGraphNet模型实现，用作对比基准
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class GraphConvLayer(nn.Module):
    """
    图卷积层，与原始HRRPGraphNet实现一致
    """

    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        xavier_uniform_(self.weight1)
        xavier_uniform_(self.weight2)
        zeros_(self.bias)

    def forward(self, x, adj):
        x = x.transpose(1, 2)
        adj = adj.squeeze(1)
        support = torch.matmul(adj, x)
        out = torch.matmul(support, self.weight1) + torch.matmul(x, self.weight2) + self.bias
        out = out.transpose(1, 2)
        return out


class HRRPGraphNet(nn.Module):
    """
    原始HRRPGraphNet模型
    """

    def __init__(self, num_classes=3, feature_dim=500):
        super(HRRPGraphNet, self).__init__()
        self.feature_dim = feature_dim

        # 1D卷积层提取局部特征
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)

        # 图卷积层捕获全局依赖
        self.graph_conv = GraphConvLayer(16, 32)

        # 非线性激活和注意力机制
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.attention = nn.Linear(feature_dim, 1)
        xavier_uniform_(self.attention.weight)
        zeros_(self.attention.bias)

        # 分类器
        self.fc = nn.Linear(feature_dim, num_classes)
        xavier_uniform_(self.fc.weight)
        zeros_(self.fc.bias)

    def construct_adj_matrix(self, x, distance_matrix):
        """构建邻接矩阵"""
        batch_size, _, N = x.size()
        x_transposed = x.transpose(1, 2)
        # 计算信号幅度相关性矩阵
        adj_matrix = torch.bmm(x_transposed, x)
        # 融合预定义的距离矩阵
        distance_matrix = distance_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        adj_matrix = adj_matrix * distance_matrix
        adj_tensor = adj_matrix.unsqueeze(1)
        return adj_tensor

    def forward(self, x, distance_matrix):
        """前向传播"""
        # 确保输入形状正确
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加通道维度

        # 构建邻接矩阵
        adj = self.construct_adj_matrix(x, distance_matrix)

        # 1D卷积提取局部特征
        x = self.leakyrelu(self.bn1(self.conv1(x)))

        # 图卷积捕获全局依赖
        x = self.graph_conv(x, adj)

        # 注意力加权
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attention_weights, dim=1)

        # 分类输出
        x = self.fc(x)
        return x

    def get_feature_extractor(self):
        """返回特征提取器部分（用于迁移学习）"""
        feature_extractor = nn.Sequential(
            self.conv1,
            self.bn1,
            self.leakyrelu
        )
        return feature_extractor