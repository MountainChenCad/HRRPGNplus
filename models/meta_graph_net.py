"""
Meta-HRRPNet主模型，集成元学习和动态图结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from models.baseline_gcn import HRRPGraphNet
from models.dynamic_gcn import DynamicGraphConv


class MetaHRRPNet(nn.Module):
    """
    Meta-HRRPNet模型，用于小样本雷达目标识别

    参数:
    - num_classes: 基础类别数量
    - feature_dim: 输入特征维度
    - hidden_dim: 隐藏层维度
    - use_dynamic_graph: 是否使用动态图生成
    - use_meta_attention: 是否使用元注意力
    - alpha: 距离衰减系数
    - num_heads: 注意力头数
    """

    def __init__(self, num_classes=5, feature_dim=500, hidden_dim=32,
                 use_dynamic_graph=True, use_meta_attention=True,
                 alpha=0.65, num_heads=4, dropout=0.1):
        super(MetaHRRPNet, self).__init__()
        self.feature_dim = feature_dim  # Keep as default but we'll adapt if needed
        self.hidden_dim = hidden_dim
        self.use_dynamic_graph = use_dynamic_graph
        self.use_meta_attention = use_meta_attention

        # 1D卷积特征提取器
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.leakyrelu = nn.LeakyReLU(0.1)

        # 图卷积层（选择动态或静态）
        if use_dynamic_graph:
            self.graph_conv = DynamicGraphConv(16, 32, hidden_dim, num_heads, alpha, dropout)
        else:
            # 使用原始HRRPGraphNet的图卷积层
            from models.baseline_gcn import GraphConvLayer
            self.graph_conv = GraphConvLayer(16, 32)

        # 初始化attention为None，将在前向传播时第一次根据实际输入尺寸进行创建
        self.attention = None
        self.use_meta_attention = use_meta_attention

        # 分类器 - 最后一层也将延迟初始化
        self.classifier_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.final_layer = None
        self.num_classes = num_classes

        # 初始化
        self._init_non_lazy_weights()

    def _init_non_lazy_weights(self):
        """初始化非延迟加载的模型权重"""
        for m in [self.conv1, self.bn1]:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _apply_attention(self, x):
        """应用注意力机制，如果需要则进行延迟初始化"""
        actual_feature_dim = x.size(2)  # Get actual feature dimension

        # 首次调用时创建注意力层
        if self.attention is None:
            if self.use_meta_attention:
                self.attention = nn.Sequential(
                    nn.Linear(actual_feature_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1)
                )
            else:
                self.attention = nn.Linear(actual_feature_dim, 1)

            # 初始化权重
            for layer in self.attention.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # 同样，需要创建最终分类层
        if self.final_layer is None:
            self.final_layer = nn.Linear(actual_feature_dim, self.num_classes)
            nn.init.xavier_uniform_(self.final_layer.weight)
            nn.init.zeros_(self.final_layer.bias)

        if self.use_meta_attention:
            # 更复杂的元注意力
            attention_logits = self.attention(x).transpose(1, 2)
            attention_weights = F.softmax(attention_logits, dim=1)
        else:
            # 原始注意力
            attention_weights = F.softmax(self.attention(x), dim=1)
        return attention_weights

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, distance_matrix=None):
        """前向传播"""
        # 确保输入形状正确
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 1D卷积提取局部特征
        x = self.leakyrelu(self.bn1(self.conv1(x)))

        # 图卷积处理
        if self.use_dynamic_graph:
            x, adj_matrix = self.graph_conv(x, distance_matrix)
        else:
            # 使用原始邻接矩阵构建方法
            adj = self._construct_static_adj(x, distance_matrix)
            x = self.graph_conv(x, adj)
            adj_matrix = adj.squeeze(1)

        # 注意力加权
        attention_weights = self._apply_attention(x)
        x = torch.sum(x * attention_weights, dim=1)

        # 在分类时，使用延迟初始化的最终层
        logits = self.classifier_layers(x)
        logits = self.final_layer(logits)

        return logits, adj_matrix

    def _construct_static_adj(self, x, distance_matrix):
        """构建静态邻接矩阵（原始方法）"""
        batch_size, _, N = x.size()
        x_transposed = x.transpose(1, 2)
        adj_matrix = torch.bmm(x_transposed, x)
        if distance_matrix is not None:
            distance_matrix = distance_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            adj_matrix = adj_matrix * distance_matrix
        adj_tensor = adj_matrix.unsqueeze(1)
        return adj_tensor

    def get_embedding(self, x, distance_matrix=None):
        """获取特征嵌入（用于可视化和分析）"""
        # 前向传播，但仅返回注意力前的特征
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.leakyrelu(self.bn1(self.conv1(x)))

        if self.use_dynamic_graph:
            x, _ = self.graph_conv(x, distance_matrix)
        else:
            adj = self._construct_static_adj(x, distance_matrix)
            x = self.graph_conv(x, adj)

        # 不应用注意力，直接返回特征
        return x