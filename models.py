import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from config import Config


class GraphAttention(nn.Module):
    """多头图注意力层"""

    def __init__(self, in_features, out_features, heads=4, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout

        # 多头注意力投影矩阵
        self.W_q = nn.Parameter(torch.FloatTensor(heads, in_features, out_features // heads))
        self.W_k = nn.Parameter(torch.FloatTensor(heads, in_features, out_features // heads))
        self.W_v = nn.Parameter(torch.FloatTensor(heads, in_features, out_features // heads))
        self.W_o = nn.Parameter(torch.FloatTensor(out_features, out_features))

        # 邻接矩阵投影
        self.W_a = nn.Parameter(torch.FloatTensor(out_features, 1))

        # 初始化参数
        xavier_uniform_(self.W_q)
        xavier_uniform_(self.W_k)
        xavier_uniform_(self.W_v)
        xavier_uniform_(self.W_o)
        xavier_uniform_(self.W_a)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x [batch_size, channels, seq_len]
        输出: attention_output [batch_size, out_features, seq_len],
              adj_matrix [batch_size, seq_len, seq_len]
        """
        batch_size, C, N = x.size()
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, channels]

        # 多头注意力
        q_heads = []
        k_heads = []
        v_heads = []

        for head in range(self.heads):
            q = torch.matmul(x_t, self.W_q[head])  # [batch_size, seq_len, out_features//heads]
            k = torch.matmul(x_t, self.W_k[head])  # [batch_size, seq_len, out_features//heads]
            v = torch.matmul(x_t, self.W_v[head])  # [batch_size, seq_len, out_features//heads]

            q_heads.append(q)
            k_heads.append(k)
            v_heads.append(v)

        # 计算注意力得分
        attn_scores = []
        attn_outputs = []

        for q, k, v in zip(q_heads, k_heads, v_heads):
            attn_score = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
            attn_score = F.softmax(attn_score, dim=-1)
            attn_score = self.dropout_layer(attn_score)
            attn_output = torch.bmm(attn_score, v)

            attn_scores.append(attn_score)
            attn_outputs.append(attn_output)

        # 合并多头注意力结果
        attn_output = torch.cat(attn_outputs, dim=-1)  # [batch_size, seq_len, out_features]
        attn_output = torch.matmul(attn_output, self.W_o)  # [batch_size, seq_len, out_features]

        # 生成邻接矩阵
        adj_matrix = torch.mean(torch.stack(attn_scores), dim=0)  # [batch_size, seq_len, seq_len]

        return attn_output.transpose(1,
                                     2), adj_matrix  # [batch_size, out_features, seq_len], [batch_size, seq_len, seq_len]


class DynamicGraphGenerator(nn.Module):
    """动态图生成模块"""

    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.1):
        super(DynamicGraphGenerator, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.attention = GraphAttention(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.lambda_mix = Config.lambda_mix

    def forward(self, x, static_adj=None):
        """
        输入: x [batch_size, in_channels, seq_len]
        输出: x_enhanced [batch_size, hidden_channels, seq_len],
              dynamic_adj [batch_size, seq_len, seq_len]
        """
        batch_size, _, seq_len = x.size()

        # 特征增强
        x_enhanced = F.leaky_relu(self.bn(self.conv1d(x)))

        # 注意力驱动的邻接矩阵生成
        _, attn_adj = self.attention(x_enhanced)

        if not Config.use_dynamic_graph:
            # 如果不使用动态图，则直接返回静态邻接矩阵
            return x_enhanced, static_adj

        if static_adj is not None:
            # 混合静态和动态邻接矩阵
            dynamic_adj = self.lambda_mix * static_adj + (1 - self.lambda_mix) * attn_adj
        else:
            dynamic_adj = attn_adj

        # 归一化邻接矩阵
        D = torch.sum(dynamic_adj, dim=2, keepdim=True)
        D_sqrt_inv = torch.pow(D, -0.5)
        D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0.0
        normalized_adj = D_sqrt_inv * dynamic_adj * D_sqrt_inv.transpose(1, 2)

        return x_enhanced, normalized_adj


class GraphConvLayer(nn.Module):
    """图卷积层"""

    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # 初始化参数
        xavier_uniform_(self.weight)
        if self.bias is not None:
            zeros_(self.bias)

    def forward(self, x, adj):
        """
        输入: x [batch_size, in_channels, seq_len]
              adj [batch_size, seq_len, seq_len]
        输出: output [batch_size, out_channels, seq_len]
        """
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, in_channels]

        # 图卷积操作
        support = torch.matmul(x_t, self.weight)  # [batch_size, seq_len, out_channels]
        output = torch.bmm(adj, support)  # [batch_size, seq_len, out_channels]

        if self.bias is not None:
            output = output + self.bias

        return output.transpose(1, 2)  # [batch_size, out_channels, seq_len]


class GraphAttentionPooling(nn.Module):
    """图注意力池化层"""

    def __init__(self, in_channels):
        super(GraphAttentionPooling, self).__init__()
        self.attention = nn.Linear(in_channels, 1)
        xavier_uniform_(self.attention.weight)
        zeros_(self.attention.bias)

    def forward(self, x):
        """
        输入: x [batch_size, in_channels, seq_len]
        输出: pooled [batch_size, in_channels]
        """
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, in_channels]
        attn_score = self.attention(x_t)  # [batch_size, seq_len, 1]
        attn_weight = F.softmax(attn_score, dim=1)  # [batch_size, seq_len, 1]

        # 加权求和
        weighted_x = x_t * attn_weight  # [batch_size, seq_len, in_channels]
        pooled = torch.sum(weighted_x, dim=1)  # [batch_size, in_channels]

        return pooled


class HRRPGraphNet(nn.Module):
    """整合动态图的HRRP图网络"""

    def __init__(self, num_classes, feature_size=None):
        super(HRRPGraphNet, self).__init__()

        # 初始化参数
        feature_size = feature_size or Config.feature_size
        hidden_channels = Config.hidden_channels
        attention_heads = Config.attention_heads
        dropout = Config.dropout

        # 动态图生成模块
        self.graph_generator = DynamicGraphGenerator(
            in_channels=1,
            hidden_channels=hidden_channels,
            heads=attention_heads,
            dropout=dropout
        )

        # 图卷积层
        self.graph_convs = nn.ModuleList()
        self.graph_convs.append(GraphConvLayer(hidden_channels, hidden_channels * 2))

        for _ in range(Config.graph_conv_layers - 1):
            self.graph_convs.append(GraphConvLayer(hidden_channels * 2, hidden_channels * 2))

        # 池化层
        self.pooling = GraphAttentionPooling(hidden_channels * 2)

        # 分类层
        self.fc = nn.Linear(hidden_channels * 2, num_classes)
        xavier_uniform_(self.fc.weight)
        zeros_(self.fc.bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static_adj=None):
        """
        输入: x [batch_size, 1, seq_len]
              static_adj [batch_size, seq_len, seq_len] 可选
        输出: logits [batch_size, num_classes]
        """
        # 生成动态图
        x_enhanced, dynamic_adj = self.graph_generator(x, static_adj)

        # 图卷积
        h = x_enhanced
        for conv in self.graph_convs:
            h = conv(h, dynamic_adj)
            h = F.leaky_relu(h)
            h = self.dropout(h)

        # 图池化
        h_pooled = self.pooling(h)

        # 分类
        logits = self.fc(h_pooled)

        return logits, dynamic_adj


class MDGN(nn.Module):
    """Meta-Dynamic Graph Network"""

    def __init__(self, num_classes=3):
        super(MDGN, self).__init__()
        self.encoder = HRRPGraphNet(num_classes=num_classes)

    def forward(self, x, static_adj=None):
        return self.encoder(x, static_adj)

    def clone(self):
        """创建模型的深拷贝，用于MAML内循环更新"""
        clone = MDGN(num_classes=self.encoder.fc.out_features)
        clone.load_state_dict(self.state_dict())
        return clone

    def adapt_params(self, loss, lr=0.01):
        """根据损失更新参数，返回更新后的参数字典"""
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        return {name: param - lr * grad for (name, param), grad in zip(self.named_parameters(), grads)}

    def set_params(self, params):
        """从参数字典设置参数"""
        for name, param in self.named_parameters():
            if name in params:
                param.data = params[name].data