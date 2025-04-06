"""
标准数据集实现，兼容原始HRRPGraphNet数据格式
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import warnings


class HRRPDataset(Dataset):
    """
    高分辨率距离像(HRRP)数据集

    参数:
    - root_dir: 包含HRRP .mat文件的目录路径
    - transform: 可选的数据变换
    - class_subset: 可选，限定加载的目标类别子集
    - random_seed: 随机种子，用于数据打乱
    """

    def __init__(self, root_dir, transform=None, class_subset=None, random_seed=3407):
        self.root_dir = root_dir
        self.transform = transform
        self.random_seed = random_seed

        # 检查目录是否存在
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"目录不存在: {root_dir}")

        # 加载所有.mat文件
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.mat')]
        if len(self.file_list) == 0:
            warnings.warn(f"在 {root_dir} 中没有找到.mat文件")

        # 提取目标类别名称
        all_classes = sorted(list(set([self.extract_target_name(f) for f in self.file_list])))

        # 如果指定了类别子集，则进行过滤
        if class_subset is not None:
            if not set(class_subset).issubset(set(all_classes)):
                invalid_classes = set(class_subset) - set(all_classes)
                raise ValueError(f"指定的类别不存在: {invalid_classes}")
            self.classes = sorted(class_subset)
            self.file_list = [f for f in self.file_list if self.extract_target_name(f) in self.classes]
        else:
            self.classes = all_classes

        # 创建类别到索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 将文件按类别分组
        self.samples_by_class = {cls: [] for cls in self.classes}
        for file_name in self.file_list:
            cls = self.extract_target_name(file_name)
            if cls in self.classes:
                self.samples_by_class[cls].append(os.path.join(self.root_dir, file_name))

        # 固定随机种子以保证可复现性
        np.random.seed(self.random_seed)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """获取单个样本"""
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        return self._load_sample(file_path)

    def _load_sample(self, file_path):
        """Load and process a single HRRP sample"""
        try:
            # Load .mat file, extract CoHH field
            data = loadmat(file_path)
            if 'CoHH' in data:
                hrrp_data = data['CoHH']
            else:
                # Try to find the first non-'__' field
                valid_keys = [k for k in data.keys() if not k.startswith('__')]
                if not valid_keys:
                    raise KeyError(f"No valid data fields found in file {file_path}")
                hrrp_data = data[valid_keys[0]]

            # Convert to PyTorch tensor
            hrrp_tensor = torch.from_numpy(abs(hrrp_data)).float()

            # Apply min-max normalization
            hrrp_min = torch.min(hrrp_tensor)
            hrrp_max = torch.max(hrrp_tensor)
            if hrrp_max > hrrp_min:
                hrrp_tensor = (hrrp_tensor - hrrp_min) / (hrrp_max - hrrp_min)

            # Apply transform (if any)
            if self.transform:
                hrrp_tensor = self.transform(hrrp_tensor)

            # Extract label
            class_name = self.extract_target_name(os.path.basename(file_path))
            label = self.class_to_idx[class_name]
            label_tensor = torch.tensor(label, dtype=torch.long)

            return hrrp_tensor, label_tensor

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, 500)), torch.tensor(-1)

    def extract_target_name(self, file_name):
        """从文件名提取目标类型"""
        return file_name.split('_')[0]

    def get_sample_by_class(self, class_idx, sample_idx=None):
        """获取指定类别的样本"""
        class_name = self.classes[class_idx]
        if sample_idx is None:
            # 随机选择一个样本
            sample_idx = np.random.randint(0, len(self.samples_by_class[class_name]))

        file_path = self.samples_by_class[class_name][sample_idx]
        return self._load_sample(file_path)

    def get_class_count(self):
        """获取每个类别的样本数量"""
        return {cls: len(files) for cls, files in self.samples_by_class.items()}

    def add_noise(self, noise_type, **params):
        """返回添加了指定噪声的数据集副本"""
        noisy_dataset = HRRPDataset(self.root_dir, transform=self.transform)
        noisy_dataset.add_noise_transform(noise_type, **params)
        return noisy_dataset

    def add_noise_transform(self, noise_type, **params):
        """添加噪声变换到当前数据集"""
        original_transform = self.transform

        # 定义噪声变换函数
        def noise_transform(x):
            # 先应用原始变换（如果有）
            if original_transform:
                x = original_transform(x)

            # 添加噪声
            if noise_type == 'gaussian':
                noise = torch.randn_like(x) * params.get('scale', 0.1)
                return x + noise
            elif noise_type == 'impulse':
                mask = torch.rand_like(x) < params.get('prob', 0.1)
                impulse = torch.randn_like(x) * params.get('strength', 1.0)
                return torch.where(mask, impulse, x)
            elif noise_type == 'speckle':
                noise = torch.randn_like(x) * params.get('scale', 0.1)
                return x * (1 + noise)
            else:
                return x

        self.transform = noise_transform