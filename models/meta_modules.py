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
        return {name: param.clone() for name, param in self.named_parameters()}

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

    def forward(self, x, params=None):
        if params is None:
            params = {k: v for k, v in self.named_parameters()}

        weight = params.get('weight', self.weight)
        bias = params.get('bias', self.bias)

        return F.batch_norm(
            x, self.running_mean, self.running_var, weight, bias,
            self.training, self.momentum, self.eps
        )


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


class MetaGraphConv(nn.Module, MetaModule):
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
        - 查询集预测结果
        """
        # 克隆初始参数
        theta = {name: param.clone() for name, param in self.model.named_parameters()}

        # 内循环适应
        for _ in range(self.inner_steps):
            logits, _ = self.model(support_x, distance_matrix)
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = torch.autograd.grad(loss, self.model.parameters(),
                                        create_graph=not self.first_order)

            # 更新参数
            updated_params = {}
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                updated_params[name] = param - self.inner_lr * grad

            # 更新模型参数
            for name, param in self.model.named_parameters():
                param.data = updated_params[name].data

        # 在查询集上预测
        query_logits, adj_matrix = self.model(query_x, distance_matrix)

        # 重置模型参数
        for name, param in self.model.named_parameters():
            param.data = theta[name].data

        return query_logits, adj_matrix


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
        # 计算原型（类别中心）
        classes = torch.unique(labels)
        n_classes = len(classes)
        prototypes = torch.zeros(n_classes, embeddings.size(1), device=embeddings.device)

        for i, c in enumerate(classes):
            mask = (labels == c)
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
        return F.cross_entropy(logits, labels)