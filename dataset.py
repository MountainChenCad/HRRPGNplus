import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import random
from config import Config


class HRRPDataset(Dataset):
    """增强的HRRP数据集，支持小样本学习"""

    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_classes = Config.all_classes if classes is None else classes
        self.labels = {target: i for i, target in enumerate(self.all_classes)}

        # 收集文件路径和标签
        self.samples = []
        for class_name in self.all_classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(class_dir, file_name)
                    label = self.labels[class_name]
                    self.samples.append((file_path, label))

        # 按类别整理样本
        self.class_samples = {}
        for class_name in self.all_classes:
            if class_name in self.labels:
                class_idx = self.labels[class_name]
                self.class_samples[class_idx] = [i for i, (_, lbl) in enumerate(self.samples) if lbl == class_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = torch.from_numpy(loadmat(file_path)['CoHH']).float()

        # 应用数据变换
        if self.transform:
            data = self.transform(data)

        return data, torch.tensor(label, dtype=torch.long)

    def get_samples_by_class(self, label, k):
        """获取指定类别的k个样本索引"""
        if label not in self.class_samples or len(self.class_samples[label]) < k:
            raise ValueError(f"Class {label} does not have enough samples (requested {k})")

        return random.sample(self.class_samples[label], k)


class HRRPTransform:
    """HRRP数据增强转换"""

    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, data):
        if not self.augment:
            return data

        # 随机选择数据增强策略
        aug_type = random.choice(['noise', 'occlusion', 'phase', 'amplitude', 'none'])

        if aug_type == 'noise':
            # 添加高斯白噪声
            snr_db = random.choice(Config.noise_levels)
            signal_power = torch.mean(torch.abs(data) ** 2)
            snr = 10 ** (snr_db / 10)
            noise_power = signal_power / snr
            noise = torch.randn_like(data) * torch.sqrt(noise_power)
            return data + noise

        elif aug_type == 'occlusion':
            # 随机遮挡
            mask = torch.ones_like(data)
            num_to_mask = int(data.shape[-1] * Config.occlusion_ratio)
            indices = torch.randperm(data.shape[-1])[:num_to_mask]
            mask[..., indices] = 0
            return data * mask

        elif aug_type == 'phase':
            # 相位抖动
            amplitude = torch.abs(data)
            phase = torch.angle(data)
            phase_noise = Config.phase_jitter * torch.randn_like(phase)
            new_phase = phase + phase_noise
            real = amplitude * torch.cos(new_phase)
            imag = amplitude * torch.sin(new_phase)
            return torch.complex(real, imag)

        elif aug_type == 'amplitude':
            # 幅度缩放
            scale = 0.8 + torch.rand(1) * 0.4  # 0.8-1.2之间随机缩放
            return data * scale

        return data


class TaskGenerator:
    """N-way K-shot任务生成器"""

    def __init__(self, dataset, n_way=None, k_shot=None, q_query=None):
        self.dataset = dataset
        self.n_way = n_way or Config.n_way
        self.k_shot = k_shot or Config.k_shot
        self.q_query = q_query or Config.q_query

        # 获取可用类别
        self.available_classes = list(dataset.class_samples.keys())
        if len(self.available_classes) < self.n_way:
            raise ValueError(
                f"Dataset has {len(self.available_classes)} classes, but {self.n_way}-way requires at least {self.n_way} classes")

    def generate_task(self):
        """生成一个N-way K-shot任务"""
        # 随机选择N个类别
        selected_classes = random.sample(self.available_classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        # 为每个类别选择支持集和查询集样本
        for i, cls in enumerate(selected_classes):
            # 获取该类别的样本索引
            samples = self.dataset.get_samples_by_class(cls, self.k_shot + self.q_query)

            # 划分支持集和查询集
            support_indices = samples[:self.k_shot]
            query_indices = samples[self.k_shot:self.k_shot + self.q_query]

            # 收集支持集样本
            for idx in support_indices:
                data, _ = self.dataset[idx]
                support_x.append(data)
                support_y.append(i)  # 使用相对标签(0到N-1)

            # 收集查询集样本
            for idx in query_indices:
                data, _ = self.dataset[idx]
                query_x.append(data)
                query_y.append(i)  # 使用相对标签(0到N-1)

        # 转换为张量
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return support_x, support_y, query_x, query_y


def prepare_datasets(scheme_idx=None):
    """准备训练和测试数据集"""
    if scheme_idx is not None:
        Config.current_scheme = scheme_idx

    # 更新n_way以匹配当前方案的类别数
    Config.update_n_way()

    scheme = Config.get_current_scheme()
    base_classes = scheme['base_classes']
    novel_classes = scheme['novel_classes']

    # 创建数据变换
    train_transform = HRRPTransform(augment=Config.augmentation)
    test_transform = HRRPTransform(augment=False)

    # 创建训练和测试数据集
    train_dataset = HRRPDataset(Config.train_dir, classes=base_classes, transform=train_transform)
    test_dataset = HRRPDataset(Config.test_dir, classes=novel_classes, transform=test_transform)

    return train_dataset, test_dataset