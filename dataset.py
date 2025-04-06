import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import random
from config import Config


class HRRPDataset(Dataset):
    """增强的HRRP数据集，支持小样本学习 - 适用于平铺数据结构"""

    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_classes = Config.all_classes if classes is None else classes
        self.labels = {target: i for i, target in enumerate(self.all_classes)}

        # 收集文件路径和标签
        self.samples = []

        # 检查目录是否存在
        if not os.path.exists(root_dir):
            print(f"警告: 数据目录不存在: {root_dir}")
            return

        # 直接从根目录加载所有.mat文件
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.mat'):
                file_path = os.path.join(root_dir, file_name)
                # 从文件名提取类别
                class_name = self.extract_class_from_filename(file_name)

                # 检查提取的类别是否在指定的类别列表中
                if class_name in self.labels:
                    label = self.labels[class_name]
                    self.samples.append((file_path, label))
                else:
                    print(f"跳过文件 {file_name}，类别 '{class_name}' 不在目标类别列表中: {self.all_classes}")

        # 打印找到的样本数量
        print(f"在 {root_dir} 中找到 {len(self.samples)} 个有效样本")

        # 按类别整理样本
        self.class_samples = {}
        for i, (_, label) in enumerate(self.samples):
            if label not in self.class_samples:
                self.class_samples[label] = []
            self.class_samples[label].append(i)

        # 打印每个类别的样本数量
        for class_name, idx in self.labels.items():
            if idx in self.class_samples:
                print(f"类别 '{class_name}' (索引 {idx}): {len(self.class_samples[idx])} 个样本")
            else:
                print(f"类别 '{class_name}' (索引 {idx}): 0 个样本")

    def extract_class_from_filename(self, filename):
        """从文件名中提取类别名称

        根据您的实际文件名格式调整此函数
        """
        # 示例: 假设文件名格式是 class_XXXX.mat 或 class-XXXX.mat
        base_name = os.path.splitext(filename)[0]

        # 尝试精确匹配类别名
        for class_name in self.all_classes:
            if class_name in base_name:
                return class_name

        # 如果没有找到匹配，解析文件名以获取类别信息
        # 可能需要根据您的文件命名约定自定义此部分
        parts = base_name.split('_')
        if len(parts) > 0:
            return parts[0]

        # 默认情况
        print(f"警告: 无法从文件名 '{filename}' 确定类别，将使用默认类别")
        return self.all_classes[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            # 加载.mat文件
            mat_data = loadmat(file_path)

            # 获取HRRP数据
            if 'hrrp' in mat_data:
                hrrp_data = mat_data['hrrp']
            else:
                # 找不到预期字段时尝试其他键
                for key in mat_data.keys():
                    if key not in ['__header__', '__version__', '__globals__']:
                        hrrp_data = mat_data[key]
                        break

            # 转换为幅度并展平
            magnitude = np.abs(hrrp_data).flatten()

            # 确保长度为500
            if len(magnitude) < 500:
                padded = np.zeros(500)
                padded[:len(magnitude)] = magnitude
                magnitude = padded
            elif len(magnitude) > 500:
                magnitude = magnitude[:500]

            # # 数据规范化 - 最小-最大缩放
            # min_val = np.min(magnitude)
            # max_val = np.max(magnitude)
            # if max_val > min_val:  # Avoid division by zero
            #     magnitude = (magnitude - min_val) / (max_val - min_val)
            # else:
            #     magnitude = np.zeros_like(magnitude)  # If all values are the same

            # 转换为张量并添加通道维度
            data = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

            # 应用数据变换
            if self.transform:
                data = self.transform(data)

            return data, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"加载文件时出错 {file_path}: {str(e)}")
            # 返回随机规范化数据
            return torch.randn(1, 500), torch.tensor(label, dtype=torch.long)

    def get_samples_by_class(self, label, k):
        """获取指定类别的k个样本索引"""
        if label not in self.class_samples or len(self.class_samples[label]) < k:
            raise ValueError(f"类别 {label} 没有足够的样本 (要求 {k}，实际 {len(self.class_samples.get(label, []))})")

        return random.sample(self.class_samples[label], k)


class HRRPTransform:
    """HRRP数据增强转换"""

    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, data):
        if not self.augment:
            return data

        # 随机选择数据增强策略，但移除phase选项，因为我们已经丢弃了相位信息
        aug_type = random.choice(['noise', 'occlusion', 'amplitude', 'none'])

        if aug_type == 'noise':
            # 添加高斯白噪声
            snr_db = random.choice(Config.noise_levels)
            signal_power = torch.mean(data ** 2)
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
                f"数据集只有 {len(self.available_classes)} 个类别, 但 {self.n_way}-way 需要至少 {self.n_way} 个类别。"
                f"可用类别: {self.available_classes}")

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

            # 收集支持集和查询集样本
            for idx in support_indices:
                data, _ = self.dataset[idx]
                support_x.append(data)
                support_y.append(i)

            for idx in query_indices:
                data, _ = self.dataset[idx]
                query_x.append(data)
                query_y.append(i)

        # 转换为张量
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        # 打印维度用于调试
        # print(f"任务形状: support_x {support_x.shape}, query_x {query_x.shape}")

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
    print(f"\n加载训练数据集: {Config.train_dir}")
    train_dataset = HRRPDataset(Config.train_dir, classes=base_classes, transform=train_transform)

    print(f"\n加载测试数据集: {Config.test_dir}")
    test_dataset = HRRPDataset(Config.test_dir, classes=novel_classes, transform=test_transform)

    return train_dataset, test_dataset