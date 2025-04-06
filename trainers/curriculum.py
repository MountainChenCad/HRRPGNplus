"""
课程学习策略实现
"""

import torch
import numpy as np
from collections import defaultdict


class CurriculumScheduler:
    """
    课程学习调度器

    参数:
    - initial_temperature: 初始温度参数
    - temperature_decay: 温度衰减系数
    - min_temperature: 最小温度值
    - task_difficulty_fn: 任务难度计算函数
    - logger: 日志记录器
    """

    def __init__(self, initial_temperature=0.5, temperature_decay=0.98,
                 min_temperature=0.01, task_difficulty_fn=None, logger=None):
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.logger = logger

        # 任务难度计算函数
        self.task_difficulty_fn = task_difficulty_fn or self._default_difficulty

        # 任务难度缓存
        self.task_difficulties = {}
        self.task_selection_history = defaultdict(int)

    def _default_difficulty(self, support_x, support_y, query_x=None, query_y=None):
        """
        默认的任务难度计算函数
        基于支持集的类内距离和类间距离比例
        """
        n_way = len(torch.unique(support_y))

        # 计算每个类别的均值向量
        class_means = {}
        for i in range(n_way):
            class_data = support_x[support_y == i]
            if len(class_data) > 0:
                # 如果是高维特征，将其展平
                if class_data.dim() > 2:
                    class_data = class_data.reshape(class_data.size(0), -1)
                class_means[i] = torch.mean(class_data, dim=0)

        # 计算类内距离和类间距离
        intra_dist = 0
        inter_dist = 0
        count_intra = 0
        count_inter = 0

        # 如果是高维特征，将其展平
        if support_x.dim() > 2:
            support_x_flat = support_x.reshape(support_x.size(0), -1)
        else:
            support_x_flat = support_x

        for i in range(len(support_x)):
            i_class = support_y[i].item()
            i_data = support_x_flat[i]

            # 计算与同类均值的距离
            if i_class in class_means:
                intra_dist += torch.norm(i_data - class_means[i_class], p=2)
                count_intra += 1

            # 计算与其他类均值的距离
            for j, mean in class_means.items():
                if j != i_class:
                    inter_dist += torch.norm(i_data - mean, p=2)
                    count_inter += 1

        # 避免除零错误
        if count_intra > 0:
            intra_dist /= count_intra
        if count_inter > 0:
            inter_dist /= count_inter

        # 难度定义：类内距离大且类间距离小的任务更难
        if inter_dist > 0:
            difficulty = intra_dist / (inter_dist + 1e-8)
        else:
            difficulty = intra_dist

        # 确保返回值是一个标量
        if isinstance(difficulty, torch.Tensor):
            difficulty = difficulty.item()
        else:
            difficulty = float(difficulty)

        return difficulty

    def compute_task_difficulty(self, task_id, support_x, support_y, query_x=None, query_y=None):
        """计算任务难度并缓存"""
        if task_id in self.task_difficulties:
            return self.task_difficulties[task_id]

        difficulty = self.task_difficulty_fn(support_x, support_y, query_x, query_y)
        self.task_difficulties[task_id] = difficulty
        return difficulty

    def update_temperature(self, epoch):
        """基于训练轮次更新温度参数"""
        self.temperature = max(
            self.initial_temperature * (self.temperature_decay ** epoch),
            self.min_temperature
        )
        return self.temperature

    def compute_sampling_weights(self, task_ids, difficulties=None):
        """计算任务采样权重"""
        if difficulties is None:
            difficulties = [self.task_difficulties.get(tid, 1.0) for tid in task_ids]

        # 使用软化的指数函数将难度转换为采样权重
        # 温度越低，对难度的敏感度越高
        weights = np.exp(-np.array(difficulties) / self.temperature)

        # 归一化为概率分布
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # 如果所有权重都为0，使用均匀分布
            weights = np.ones_like(weights) / len(weights)

        return weights

    def sample_task_indices(self, num_tasks, task_ids, difficulties=None):
        """基于当前温度和任务难度采样任务索引"""
        weights = self.compute_sampling_weights(task_ids, difficulties)

        # 根据权重采样任务
        indices = np.random.choice(len(task_ids), size=num_tasks, p=weights)

        # 更新任务选择历史
        for idx in indices:
            task_id = task_ids[idx]
            self.task_selection_history[task_id] += 1

        return indices

    def get_temperature(self):
        """获取当前温度参数"""
        return self.temperature

    def get_sampling_statistics(self):
        """获取任务采样统计信息"""
        total_selections = sum(self.task_selection_history.values())
        if total_selections == 0:
            return {}

        # 按难度分组计算选择概率
        difficulty_groups = {
            'easy': [],
            'medium': [],
            'hard': []
        }

        # 对任务按难度分组
        difficulties = list(self.task_difficulties.values())
        if difficulties:
            # 使用百分位数进行分组
            if len(difficulties) >= 3:
                q1, q2 = np.percentile(difficulties, [33, 66])
            else:
                # 当样本量太小时使用简单的划分
                sorted_diffs = sorted(difficulties)
                n = len(sorted_diffs)
                q1 = sorted_diffs[max(0, n // 3 - 1)]
                q2 = sorted_diffs[min(n - 1, 2 * n // 3)]

            for task_id, diff in self.task_difficulties.items():
                if diff <= q1:
                    group = 'easy'
                elif diff <= q2:
                    group = 'medium'
                else:
                    group = 'hard'

                if task_id in self.task_selection_history:
                    difficulty_groups[group].append(self.task_selection_history[task_id])

            # 计算每个难度组的平均选择频率
            group_stats = {}
            for group, counts in difficulty_groups.items():
                if counts:
                    group_stats[group] = {
                        'count': len(counts),
                        'avg_selection': sum(counts) / len(counts),
                        'selection_ratio': sum(counts) / total_selections if total_selections > 0 else 0
                    }

            return group_stats

        return {}