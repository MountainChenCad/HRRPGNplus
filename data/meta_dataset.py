"""
元学习数据集 - 实现N-way K-shot任务采样
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from data.dataset import HRRPDataset
import random


class TaskSampler:
    """
    任务采样器 - 从基础数据集随机采样N-way K-shot任务

    参数:
    - dataset: HRRPDataset实例
    - n_way: 每个任务中的类别数
    - k_shot: 每个类别的支持集样本数
    - q_query: 每个类别的查询集样本数
    - num_tasks: 总共生成的任务数
    - fixed_tasks: 是否固定任务集（测试用）
    """

    def __init__(self, dataset, n_way, k_shot, q_query, num_tasks=1000, fixed_tasks=False, seed=None):
        self.dataset = dataset
        self.n_way = min(n_way, len(dataset.classes))
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        self.fixed_tasks = fixed_tasks

        # 设置随机种子
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # 检查每个类别是否有足够的样本
        class_counts = dataset.get_class_count()
        for cls, count in class_counts.items():
            if count < k_shot + q_query:
                raise ValueError(f"类别 {cls} 只有 {count} 个样本，少于所需的 {k_shot + q_query} 个")

        # 如果固定任务集，预生成所有任务
        if fixed_tasks:
            self.tasks = self._generate_all_tasks()

    def _generate_all_tasks(self):
        """预生成所有任务"""
        tasks = []
        for _ in range(self.num_tasks):
            tasks.append(self._sample_task())
        return tasks

    def _sample_task(self):
        """采样单个任务（N-way, K-shot, Q-query）"""
        # 随机选择N个类别
        selected_classes = np.random.choice(len(self.dataset.classes), self.n_way, replace=False)

        # 为每个类别采样K+Q个样本
        support_set = []
        query_set = []

        for class_idx in selected_classes:
            # 获取当前类别的所有样本索引
            class_name = self.dataset.classes[class_idx]
            sample_indices = list(range(len(self.dataset.samples_by_class[class_name])))

            # 随机选择K+Q个不重复的样本
            selected_indices = np.random.choice(sample_indices, self.k_shot + self.q_query, replace=False)

            # 前K个用于支持集
            for idx in selected_indices[:self.k_shot]:
                data, _ = self.dataset.get_sample_by_class(class_idx, idx)
                support_set.append((data, class_idx))

            # 后Q个用于查询集
            for idx in selected_indices[self.k_shot:]:
                data, _ = self.dataset.get_sample_by_class(class_idx, idx)
                query_set.append((data, class_idx))

        # 打乱支持集和查询集
        random.shuffle(support_set)
        random.shuffle(query_set)

        return {
            'support_x': torch.stack([item[0] for item in support_set]),
            'support_y': torch.tensor([item[1] for item in support_set]),
            'query_x': torch.stack([item[0] for item in query_set]),
            'query_y': torch.tensor([item[1] for item in query_set]),
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }

    def sample(self, batch_size=1):
        """采样一批任务"""
        if self.fixed_tasks:
            # 从预生成的任务中随机选择
            indices = np.random.choice(self.num_tasks, batch_size)
            batch = [self.tasks[i] for i in indices]
        else:
            # 动态生成任务
            batch = [self._sample_task() for _ in range(batch_size)]

        # 合并批次中的所有任务
        support_x = torch.stack([task['support_x'] for task in batch])
        support_y = torch.stack([task['support_y'] for task in batch])
        query_x = torch.stack([task['query_x'] for task in batch])
        query_y = torch.stack([task['query_y'] for task in batch])

        return {
            'support_x': support_x,  # [batch_size, n_way*k_shot, ...]
            'support_y': support_y,  # [batch_size, n_way*k_shot]
            'query_x': query_x,  # [batch_size, n_way*q_query, ...]
            'query_y': query_y,  # [batch_size, n_way*q_query]
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }


class MetaHRRPDataset(Dataset):
    """
    元学习HRRP数据集，封装TaskSampler为PyTorch Dataset接口

    参数:
    - dataset: 基础HRRP数据集
    - n_way: 分类任务的类别数
    - k_shot: 支持集中每类样本数
    - q_query: 查询集中每类样本数
    - num_tasks: 数据集中的任务总数
    - task_augment: 是否对任务进行数据增强
    """

    def __init__(self, dataset, n_way, k_shot, q_query, num_tasks=1000, task_augment=False):
        self.dataset = dataset
        self.sampler = TaskSampler(
            dataset, n_way, k_shot, q_query, num_tasks, fixed_tasks=True
        )
        self.task_augment = task_augment

    # Forward all necessary attributes to the base dataset
    @property
    def classes(self):
        return self.dataset.classes

    @property
    def samples_by_class(self):
        return self.dataset.samples_by_class

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    @property
    def root_dir(self):
        return self.dataset.root_dir

    def get_class_count(self):
        return self.dataset.get_class_count()

    def get_sample_by_class(self, class_idx, sample_idx=None):
        return self.dataset.get_sample_by_class(class_idx, sample_idx)

    def extract_target_name(self, file_name):
        return self.dataset.extract_target_name(file_name)

    def __len__(self):
        return self.sampler.num_tasks

    def __getitem__(self, idx):
        """获取指定索引的任务"""
        task = self.sampler.tasks[idx]

        if self.task_augment:
            # 对支持集和查询集应用相同的增强
            task = self._augment_task(task)

        return task

    def _augment_task(self, task):
        """对任务进行数据增强"""
        # 随机旋转（模拟不同视角）
        if random.random() < 0.5:
            task['support_x'] = self._random_shift(task['support_x'])
            task['query_x'] = self._random_shift(task['query_x'])

        # 随机添加轻微噪声
        if random.random() < 0.3:
            noise_scale = random.uniform(0.01, 0.05)
            task['support_x'] += torch.randn_like(task['support_x']) * noise_scale
            task['query_x'] += torch.randn_like(task['query_x']) * noise_scale

        return task

    def _random_shift(self, x, max_shift=10):
        """随机位移模拟距离偏移"""
        batch_size = x.size(0)
        shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,))

        # 对每个样本应用不同的位移
        shifted_x = torch.zeros_like(x)
        for i, shift in enumerate(shifts):
            if shift == 0:
                shifted_x[i] = x[i]
            elif shift > 0:
                shifted_x[i, :, shift:] = x[i, :, :-shift]
            else:
                shifted_x[i, :, :shift] = x[i, :, -shift:]

        return shifted_x


class CurriculumTaskSampler(TaskSampler):
    """
    基于课程学习的任务采样器

    参数:
    - 与TaskSampler相同
    - temperature: 温度参数，控制任务采样概率
    - temp_decay: 温度衰减率
    """

    def __init__(self, dataset, n_way, k_shot, q_query, num_tasks=1000,
                 fixed_tasks=False, seed=None, temperature=0.5, temp_decay=0.98):
        super().__init__(dataset, n_way, k_shot, q_query, num_tasks, fixed_tasks, seed)

        self.temperature = temperature
        self.init_temp = temperature
        self.temp_decay = temp_decay

        # 初始化任务难度
        if fixed_tasks:
            self.task_difficulties = self._compute_task_difficulties()

    def _compute_task_difficulties(self):
        """计算所有任务的难度"""
        difficulties = []
        for task in self.tasks:
            # 提取支持集数据
            support_x = task['support_x']  # [n_way*k_shot, ...]
            support_y = task['support_y']  # [n_way*k_shot]

            # 计算每个类别的均值向量
            class_means = {}
            for i in range(self.n_way):
                class_data = support_x[support_y == i]
                if len(class_data) > 0:
                    class_means[i] = torch.mean(class_data, dim=0)

            # 计算类内距离和类间距离
            intra_dist = 0
            inter_dist = 0
            count_intra = 0
            count_inter = 0

            for i in range(len(support_x)):
                i_class = support_y[i].item()
                i_data = support_x[i]

                # 计算与同类均值的距离
                if i_class in class_means:
                    intra_dist += torch.norm(i_data - class_means[i_class], p=2)
                    count_intra += 1

                # 计算与其他类均值的距离
                for j, mean in class_means.items():
                    if j != i_class:
                        inter_dist += torch.norm(i_data - mean, p=2)
                        count_inter += 1

            if count_intra > 0:
                intra_dist /= count_intra
            if count_inter > 0:
                inter_dist /= count_inter

            # 难度定义：类内距离大且类间距离小的任务更难
            if inter_dist > 0:
                difficulty = intra_dist / (inter_dist + 1e-8)
            else:
                difficulty = intra_dist

            # To this:
            if isinstance(difficulty, torch.Tensor):
                difficulties.append(difficulty.item())
            else:
                difficulties.append(float(difficulty))

        return np.array(difficulties)

    def update_temperature(self, epoch):
        """更新温度参数"""
        self.temperature = self.init_temp * (self.temp_decay ** epoch)
        return self.temperature

    def sample(self, batch_size=1):
        """基于课程学习采样任务"""
        if not self.fixed_tasks:
            # 动态生成任务时，直接使用父类实现
            return super().sample(batch_size)

        # 根据难度和当前温度计算任务采样概率
        probs = np.exp(-self.task_difficulties / self.temperature)
        probs = probs / np.sum(probs)  # 归一化

        # 根据概率采样任务
        indices = np.random.choice(self.num_tasks, batch_size, p=probs)
        batch = [self.tasks[i] for i in indices]

        # 合并批次中的所有任务
        support_x = torch.stack([task['support_x'] for task in batch])
        support_y = torch.stack([task['support_y'] for task in batch])
        query_x = torch.stack([task['query_x'] for task in batch])
        query_y = torch.stack([task['query_y'] for task in batch])

        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y,
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'task_indices': indices  # 额外返回采样的任务索引，用于分析
        }