"""
动态图卷积模块，支持自适应邻接矩阵生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class DynamicGraphAttention(nn.Module):
    """
    动态图注意力机制，学习自适应邻接矩阵
    """

    def __init__(self, in_channels, hidden_dim=32, num_heads=4, alpha=0.65, dropout=0.1):
        super(DynamicGraphAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.alpha = nn.Parameter(torch.tensor(alpha))  # 可学习的距离衰减系数
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 多头查询-键-值投影
        self.q_proj = nn.Linear(in_channels, hidden_dim)
        self.k_proj = nn.Linear(in_channels, hidden_dim)
        self.v_proj = nn.Linear(in_channels, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_channels)

        # 初始化
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.out_proj.weight)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """重置可学习参数"""
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x, distance_matrix=None):
        """
        前向传播

        参数:
        - x: 输入特征 [batch_size, channels, seq_len]
        - distance_matrix: 预定义距离矩阵 [seq_len, seq_len]

        返回:
        - 输出特征和动态邻接矩阵
        """
        batch_size, channels, seq_len = x.size()

        # 特征转置为 [batch_size, seq_len, channels]
        x_t = x.transpose(1, 2)

        # 多头注意力投影
        q = self.q_proj(x_t).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_t).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x_t).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # 调整形状为 [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 计算注意力得分
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用距离衰减矩阵（如果提供）
        if distance_matrix is not None:
            # 使用可学习的衰减系数alpha调整距离影响
            distance_influence = 1.0 / (self.alpha * distance_matrix + 1.0)
            distance_influence = distance_influence.unsqueeze(0).unsqueeze(1)
            attn_weights = attn_weights * distance_influence

        # 归一化注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        out = torch.matmul(attn_weights, v)

        # 形状变回 [batch_size, seq_len, hidden_dim]
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)

        # 最终投影
        out = self.out_proj(out)

        # 转回原始形状 [batch_size, channels, seq_len]
        out = out.transpose(1, 2)

        # 返回输出特征和动态构建的邻接矩阵
        return out, attn_weights.mean(dim=1)  # 平均多头注意力权重作为邻接矩阵


class DynamicGraphConv(nn.Module):
    """
    动态图卷积层，集成自适应邻接矩阵生成
    """

    def __init__(self, in_channels, out_channels, hidden_dim=32, num_heads=4, alpha=0.65, dropout=0.1):
        super(DynamicGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 动态图注意力模块
        self.graph_attention = DynamicGraphAttention(
            in_channels, hidden_dim, num_heads, alpha, dropout
        )

        # 图卷积权重
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

        # 初始化
        xavier_uniform_(self.weight)
        zeros_(self.bias)

    def forward(self, x, distance_matrix=None):
        """
        前向传播

        参数:
        - x: 输入特征 [batch_size, in_channels, seq_len]
        - distance_matrix: 预定义距离矩阵 [seq_len, seq_len]

        返回:
        - 输出特征 [batch_size, out_channels, seq_len]
        - 动态邻接矩阵
        """
        # 获取动态邻接矩阵和中间特征
        h_attn, adj_matrix = self.graph_attention(x, distance_matrix)

        # 图卷积操作
        batch_size, _, seq_len = x.size()
        h_attn_t = h_attn.transpose(1, 2)  # [batch_size, seq_len, in_channels]

        # 邻接矩阵应用
        support = torch.bmm(adj_matrix, h_attn_t)  # [batch_size, seq_len, in_channels]

        # 权重变换
        out = torch.matmul(support, self.weight) + self.bias  # [batch_size, seq_len, out_channels]

        # 转置回原始形状
        out = out.transpose(1, 2)  # [batch_size, out_channels, seq_len]

        # 残差连接
        res = self.residual(x)
        out = out + res

        return out, adj_matrix