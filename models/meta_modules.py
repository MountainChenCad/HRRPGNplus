"""
元学习模块实现，包括MAML组件和适应性层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy


class MetaModule(nn.Module):
    """
    元学习基础模块，定义一些通用方法
    """

    def __init__(self):
        super(MetaModule, self).__init__()

    def clone_parameters(self):
        """深拷贝模型参数并返回"""
        return OrderedDict({name: param.clone() for name, param in self.named_parameters()})

    def clone_with_weights(self, weights_dict):
        """
        创建一个带有指定权重的模型副本

        参数:
        - weights_dict: 权重字典

        返回:
        - 带有新权重的模型副本
        """
        clone = deepcopy(self)
        for name, param in clone.named_parameters():
            if name in weights_dict:
                param.data = weights_dict[name].data.to(param.device)
        return clone

    def point_grad_to(self, target):
        """
        将当前模块的梯度指向目标模块
        用于MAML的二阶导数计算
        """
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if target_p.grad is not None:
                    p.grad = target_p.grad.clone()
            else:
                if target_p.grad is None:
                    target_p.grad = p.grad.clone()
                else:
                    target_p.grad.data.copy_(p.grad.data)

    def update_params(self, lr, grads=None):
        """
        使用给定的梯度更新参数

        参数:
        - lr: 学习率
        - grads: 参数的梯度字典，如果为None则使用.grad属性

        返回:
        - 更新后的模型参数字典
        """
        if grads is None:
            grads = {name: param.grad for name, param in self.named_parameters()}

        updated_params = OrderedDict()

        for name, param in self.named_parameters():
            if grads[name] is None:
                updated_params[name] = param
            else:
                updated_params[name] = param - lr * grads[name]

        return updated_params

    def transfer_weights(self, weights_dict):
        """
        将指定的权重转移到当前模型

        参数:
        - weights_dict: 权重字典
        """
        for name, param in self.named_parameters():
            if name in weights_dict:
                param.data = weights_dict[name].data.to(param.device)


class MetaConv1d(nn.Conv1d, MetaModule):
    """元卷积1D层"""

    def __init__(self, *args, **kwargs):
        super(MetaConv1d, self).__init__(*args, **kwargs)

    def forward(self, x, params=None):
        if params is None:
            params = {k: v for k, v in self.named_parameters()}

        weight = params.get('weight', self.weight)
        bias = params.get('bias', self.bias)

        return F.conv1d(
            x, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )


class MetaLinear(nn.Linear, MetaModule):
    """元线性层"""

    def __init__(self, *args, **kwargs):
        super(MetaLinear, self).__init__(*args, **kwargs)

    def forward(self, x, params=None):
        if params is None:
            params = {k: v for k, v in self.named_parameters()}

        weight = params.get('weight', self.weight)
        bias = params.get('bias', self.bias)

        return F.linear(x, weight, bias)


class MetaBatchNorm1d(nn.BatchNorm1d, MetaModule):
    """元批量归一化1D层"""

    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm1d, self).__init__(*args, **kwargs)

        # Buffers need special handling for meta-learning
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x, params=None):
        if params is None:
            params = {k: v for k, v in self.named_parameters()}

        weight = params.get('weight', self.weight)
        bias = params.get('bias', self.bias)

        if self.training:
            # During training, calculate stats but use the provided ones
            _, running_var, running_mean = torch.var_mean(
                x, dim=[0, 2], unbiased=False, keepdim=True
            )
            # Update running stats for evaluation mode
            self.running_mean = self.running_mean * (1 - self.momentum) + running_mean.reshape(-1) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + running_var.reshape(-1) * self.momentum
            self.num_batches_tracked += 1
            # Normalize with current batch stats
            norm_x = (x - running_mean) / torch.sqrt(running_var + self.eps)
        else:
            # In eval mode, use running stats
            norm_x = (x - self.running_mean.view(1, -1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1) + self.eps)

        # Apply gamma and beta
        return weight.view(1, -1, 1) * norm_x + bias.view(1, -1, 1)


class MetaSequential(nn.Sequential, MetaModule):
    """元序列容器"""

    def __init__(self, *args):
        super(MetaSequential, self).__init__(*args)

    def forward(self, x, params=None):
        if params is None:
            params = {k: v for k, v in self.named_parameters()}

        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                module_params = {k.replace(name + '.', ''): v for k, v in params.items()
                                 if k.startswith(name)}
                x = module(x, module_params)
            else:
                x = module(x)

        return x


class MetaGraphConv(MetaModule):
    """元图卷积层"""

    def __init__(self, in_channels, out_channels):
        super(MetaGraphConv, self).__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, params=None):
        if params is None:
            params = {k: v for k, v in self.named_parameters()}

        weight1 = params.get('weight1', self.weight1)
        weight2 = params.get('weight2', self.weight2)
        bias = params.get('bias', self.bias)

        x = x.transpose(1, 2)
        adj = adj.squeeze(1)
        support = torch.matmul(adj, x)
        out = torch.matmul(support, weight1) + torch.matmul(x, weight2) + bias
        out = out.transpose(1, 2)
        return out


class MAML(nn.Module):
    """
    模型无关元学习（MAML）实现

    参数:
    - model: 基础模型（应该是MetaModule类型）
    - inner_lr: 内循环学习率
    - inner_steps: 内循环更新步数
    - first_order: 是否使用一阶近似
    """

    def __init__(self, model, inner_lr=0.01, inner_steps=5, first_order=False):
        super(MAML, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

    def forward(self, support_x, support_y, query_x, distance_matrix=None):
        """
        MAML前向传播

        参数:
        - support_x: 支持集输入
        - support_y: 支持集标签
        - query_x: 查询集输入
        - distance_matrix: 距离矩阵（如有）

        返回:
        - query_logits: 查询集预测结果
        - adj_matrix: 邻接矩阵
        """
        device = support_x.device

        # 克隆初始参数
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # 内循环适应
        for _ in range(self.inner_steps):
            # 前向传播
            logits, adj_matrix = self.model(support_x, distance_matrix)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                self.model.parameters(),
                create_graph=not self.first_order,
                retain_graph=not self.first_order
            )

            # 更新参数
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        # 在查询集上预测
        query_logits, adj_matrix = self.model(query_x, distance_matrix)

        # 重置模型参数
        for name, param in self.model.named_parameters():
            param.data = original_params[name].data

        return query_logits, adj_matrix

    def adapt(self, support_x, support_y, distance_matrix=None, steps=None):
        """
        适应新任务

        参数:
        - support_x: 支持集输入
        - support_y: 支持集标签
        - distance_matrix: 距离矩阵（如有）
        - steps: 适应步数，默认使用self.inner_steps

        返回:
        - adapted_model: 适应后的模型
        """
        if steps is None:
            steps = self.inner_steps

        adapted_model = deepcopy(self.model)

        # 内循环适应
        for _ in range(steps):
            # 前向传播
            logits, _ = adapted_model(support_x, distance_matrix)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=False,
                retain_graph=False
            )

            # 更新参数
            for (name, param), grad in zip(adapted_model.named_parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        return adapted_model


class ProtoLoss(nn.Module):
    """
    原型网络损失，用于增强小样本学习能力

    参数:
    - n_way: 任务类别数
    - k_shot: 每类样本数
    - metric: 距离度量方式，'euclidean'或'cosine'
    """

    def __init__(self, n_way=5, k_shot=1, metric='euclidean'):
        super(ProtoLoss, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.metric = metric

    def forward(self, embeddings, labels):
        """
        计算原型损失

        参数:
        - embeddings: 样本嵌入特征 [batch_size, feature_dim]
        - labels: 样本标签 [batch_size]

        返回:
        - 原型损失值
        """
        # 确保输入在正确设备上
        device = embeddings.device

        # 计算原型（类别中心）
        classes = torch.unique(labels)
        n_classes = len(classes)

        if n_classes <= 1:
            # 如果只有一个类别，则无法计算对比损失
            return torch.tensor(0.0, device=device)

        prototypes = torch.zeros(n_classes, embeddings.size(1), device=device)

        for i, c in enumerate(classes):
            mask = (labels == c)
            if mask.sum() > 0:  # Ensure we have samples for this class
                prototypes[i] = embeddings[mask].mean(dim=0)

        # 计算每个样本到各原型的距离
        if self.metric == 'euclidean':
            # 欧氏距离
            distances = torch.cdist(embeddings, prototypes) ** 2
        elif self.metric == 'cosine':
            # 余弦相似度（转换为距离）
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(embeddings_norm, prototypes_norm.t())
            distances = 1 - similarities
        else:
            raise ValueError(f"不支持的度量类型: {self.metric}")

        # 计算交叉熵损失
        logits = -distances

        # 确保标签在有效范围内
        label_indices = torch.zeros_like(labels)
        for i, c in enumerate(classes):
            label_indices[labels == c] = i

        return F.cross_entropy(logits, label_indices)