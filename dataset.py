import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.io import loadmat
import random
import scipy.signal as signal
from config import Config
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

# Define CVPR-quality color palette
COLORS = ['#0783D5', '#E52119', '#FD751F', '#0E2D88', '#78196D',
          '#C2C121', '#FC837E', '#00A6BC', '#025057', '#7E5505', '#77196C']

# Set global matplotlib parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12


class HRRPDataset(Dataset):
    """增强的HRRP数据集，支持小样本学习 - 适用于平铺数据结构"""

    def __init__(self, root_dir, classes=None, transform=None, dataset_type='simulated'):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_type = dataset_type
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
            if file_name.endswith('.mat') or file_name.endswith('.h5'):
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
            if class_name.lower() in base_name.lower():
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
            # 根据文件类型加载数据
            if file_path.endswith('.mat'):
                hrrp_data = self._load_mat_file(file_path)
            elif file_path.endswith('.h5'):
                hrrp_data = self._load_h5_file(file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_path}")

            # 转换为幅度并展平
            magnitude = np.abs(hrrp_data).flatten()

            # 确保长度为500
            if len(magnitude) < Config.feature_size:
                padded = np.zeros(Config.feature_size)
                padded[:len(magnitude)] = magnitude
                magnitude = padded
            elif len(magnitude) > Config.feature_size:
                magnitude = magnitude[:Config.feature_size]

            # 转换为张量并添加通道维度
            data = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)

            # 应用数据变换
            if self.transform:
                data = self.transform(data)

            return data, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"加载文件时出错 {file_path}: {str(e)}")
            # 返回随机规范化数据
            return torch.randn(1, Config.feature_size), torch.tensor(label, dtype=torch.long)

    def _load_mat_file(self, file_path):
        """加载MATLAB .mat文件"""
        try:
            # 尝试使用scipy.io.loadmat加载
            mat_data = loadmat(file_path)

            # 获取HRRP数据
            if 'hrrp' in mat_data:
                return mat_data['hrrp']
            else:
                # 找不到预期字段时尝试其他键
                for key in mat_data.keys():
                    if key not in ['__header__', '__version__', '__globals__']:
                        return mat_data[key]

        except NotImplementedError:
            # 如果scipy.io.loadmat失败，尝试使用h5py
            try:
                with h5py.File(file_path, 'r') as f:
                    # 尝试常见的键名称
                    for key in ['hrrp', 'data', 'x']:
                        if key in f:
                            return np.array(f[key][()])

                    # 如果没有找到预期键，使用第一个数据集
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            return np.array(f[key][()])
            except:
                raise ValueError(f"无法加载文件: {file_path}")

        raise ValueError(f"在文件中找不到HRRP数据: {file_path}")

    def _load_h5_file(self, file_path):
        """加载HDF5 .h5文件"""
        with h5py.File(file_path, 'r') as f:
            # 尝试常见的键名称
            for key in ['hrrp', 'data', 'x']:
                if key in f:
                    return np.array(f[key][()])

            # 如果没有找到预期键，使用第一个数据集
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    return np.array(f[key][()])

        raise ValueError(f"在H5文件中找不到HRRP数据: {file_path}")

    def get_samples_by_class(self, label, k):
        """获取指定类别的k个样本索引"""
        if label not in self.class_samples or len(self.class_samples[label]) < k:
            raise ValueError(f"类别 {label} 没有足够的样本 (要求 {k}，实际 {len(self.class_samples.get(label, []))})")

        return random.sample(self.class_samples[label], k)

    def get_class_distribution(self):
        """获取数据集的类别分布"""
        class_counts = {}
        for class_name, idx in self.labels.items():
            if idx in self.class_samples:
                class_counts[class_name] = len(self.class_samples[idx])
            else:
                class_counts[class_name] = 0

        return class_counts

    def visualize_samples(self, num_samples=5, save_path=None):
        """可视化数据集样本"""
        plt.figure(figsize=(15, 10))

        # 为每个类别选择样本
        for class_idx, class_name in enumerate(self.all_classes):
            if class_idx not in self.class_samples or not self.class_samples[class_idx]:
                continue

            # 随机选择该类别的样本
            sample_indices = random.sample(self.class_samples[class_idx],
                                           min(num_samples, len(self.class_samples[class_idx])))

            for i, sample_idx in enumerate(sample_indices):
                # 获取样本
                data, _ = self[sample_idx]

                # 转换为numpy数组
                hrrp_data = data.squeeze().numpy()

                # 绘制样本
                plt.subplot(len(self.all_classes), num_samples,
                            class_idx * num_samples + i + 1)
                plt.plot(hrrp_data)
                plt.title(f"{class_name} - Sample {i + 1}")
                plt.grid(True)

                if i == 0:
                    plt.ylabel("Amplitude")

                if class_idx == len(self.all_classes) - 1:
                    plt.xlabel("Range Cell")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class HRRPTransform:
    """HRRP数据增强转换"""

    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, data):
        if not self.augment:
            return data

        # 随机选择数据增强策略
        aug_probs = {
            'noise': 0.3,
            'occlusion': 0.2,
            'amplitude': 0.2,
            'shift': 0.1,
            'none': 0.2
        }

        aug_type = np.random.choice(
            list(aug_probs.keys()),
            p=list(aug_probs.values())
        )

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

        elif aug_type == 'shift':
            # 随机位移
            shift_amount = int(data.shape[-1] * 0.05)  # 最多移动5%
            shift = random.randint(-shift_amount, shift_amount)

            if shift > 0:
                # 右移
                shifted_data = torch.zeros_like(data)
                shifted_data[..., shift:] = data[..., :-shift]
                return shifted_data
            elif shift < 0:
                # 左移
                shifted_data = torch.zeros_like(data)
                shifted_data[..., :shift] = data[..., -shift:]
                return shifted_data

        return data

    def add_noise(self, data, snr_db):
        """添加指定信噪比的噪声"""
        signal_power = torch.mean(data ** 2)
        snr = 10 ** (snr_db / 10)
        noise_power = signal_power / snr
        noise = torch.randn_like(data) * torch.sqrt(noise_power)
        return data + noise

    def add_occlusion(self, data, occlusion_ratio=None):
        """添加随机遮挡"""
        if occlusion_ratio is None:
            occlusion_ratio = Config.occlusion_ratio

        mask = torch.ones_like(data)
        num_to_mask = int(data.shape[-1] * occlusion_ratio)
        indices = torch.randperm(data.shape[-1])[:num_to_mask]
        mask[..., indices] = 0
        return data * mask

    def add_phase_jitter(self, data, jitter_strength=None):
        """添加相位抖动"""
        if jitter_strength is None:
            jitter_strength = Config.phase_jitter

        # 将数据视为复数信号进行抖动
        # 注意：此操作主要适用于复信号，对实信号效果有限
        phase_noise = torch.rand_like(data) * 2 * jitter_strength - jitter_strength
        return data * torch.exp(1j * phase_noise)

    def add_random_shift(self, data, max_shift_ratio=0.05):
        """添加随机位移"""
        shift_amount = int(data.shape[-1] * max_shift_ratio)
        shift = random.randint(-shift_amount, shift_amount)

        if shift == 0:
            return data

        shifted_data = torch.zeros_like(data)
        if shift > 0:
            # 右移
            shifted_data[..., shift:] = data[..., :-shift]
        else:
            # 左移
            shifted_data[..., :shift] = data[..., -shift:]

        return shifted_data


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

        return support_x, support_y, query_x, query_y

    def generate_fixed_task(self, fixed_classes=None):
        """生成一个固定类别的N-way K-shot任务"""
        if fixed_classes is None or len(fixed_classes) < self.n_way:
            raise ValueError(f"需要至少 {self.n_way} 个有效类别来生成固定任务")

        # 确保所有类别存在于数据集中
        valid_classes = []
        for cls in fixed_classes:
            if cls in self.dataset.class_samples:
                valid_classes.append(cls)
            else:
                print(f"警告: 类别 {cls} 在数据集中不存在")

        if len(valid_classes) < self.n_way:
            raise ValueError(f"没有足够的有效类别来生成固定任务")

        # 选择N个类别
        selected_classes = valid_classes[:self.n_way]

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

        return support_x, support_y, query_x, query_y

    def generate_noisy_task(self, snr_db):
        """生成一个带噪声的N-way K-shot任务"""
        # 生成标准任务
        support_x, support_y, query_x, query_y = self.generate_task()

        # 创建噪声变换
        noise_transform = HRRPTransform(augment=False)

        # 向查询集添加噪声
        noisy_query_x = []
        for x in query_x:
            noisy_x = noise_transform.add_noise(x.unsqueeze(0), snr_db).squeeze(0)
            noisy_query_x.append(noisy_x)

        # 转换为张量
        noisy_query_x = torch.stack(noisy_query_x)

        return support_x, support_y, noisy_query_x, query_y


def prepare_datasets(scheme_idx=None, dataset_type='measured'):
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
    train_dataset = HRRPDataset(Config.train_dir, classes=base_classes,
                                transform=train_transform, dataset_type=dataset_type)

    print(f"\n加载测试数据集: {Config.test_dir}")
    test_dataset = HRRPDataset(Config.test_dir, classes=novel_classes,
                               transform=test_transform, dataset_type=dataset_type)

    return train_dataset, test_dataset


def visualize_dataset_statistics(dataset, save_path=None):
    """可视化数据集统计信息"""
    class_dist = dataset.get_class_distribution()

    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[1, 1])

    # 绘制类别分布
    ax1 = plt.subplot(gs[0])
    class_names = list(class_dist.keys())
    counts = list(class_dist.values())

    # Use colors from the palette
    bars = ax1.bar(range(len(class_names)), counts,
                   color=[COLORS[i % len(COLORS)] for i in range(len(class_names))],
                   width=0.7, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)

    # Use integer y-axis values
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 绘制样本频谱
    ax2 = plt.subplot(gs[1])

    # 随机选择每个类的一个样本
    class_samples = []
    class_names = []

    for class_name, class_idx in dataset.labels.items():
        if class_idx in dataset.class_samples and len(dataset.class_samples[class_idx]) > 0:
            sample_idx = random.choice(dataset.class_samples[class_idx])
            data, _ = dataset[sample_idx]
            class_samples.append(data.squeeze().numpy())
            class_names.append(class_name)

    # 计算和绘制频谱
    for i, (sample, name) in enumerate(zip(class_samples, class_names)):
        # 计算频谱
        fft_result = np.abs(np.fft.fft(sample))
        fft_result = fft_result[:len(fft_result) // 2]  # 只保留正频率部分

        ax2.plot(fft_result, color=COLORS[i % len(COLORS)], linewidth=2.5, label=name)

    ax2.set_title('Sample Frequency Spectrum', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='major', width=1.5, length=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_data_augmentation(dataset, save_path=None):
    """可视化数据增强效果"""
    # 随机选择一个样本
    sample_idx = random.choice(range(len(dataset)))
    original_data, label = dataset[sample_idx]
    class_name = dataset.all_classes[label]

    # 创建增强器
    augmenter = HRRPTransform(augment=True)

    # 应用不同类型的增强
    aug_types = {
        'Original': original_data,
        'Noise (SNR=10dB)': augmenter.add_noise(original_data, 10),
        'Noise (SNR=0dB)': augmenter.add_noise(original_data, 0),
        'Occlusion (10%)': augmenter.add_occlusion(original_data, 0.1),
        'Occlusion (20%)': augmenter.add_occlusion(original_data, 0.2),
        'Shift Right': augmenter.add_random_shift(original_data)
    }

    # 可视化
    fig = plt.figure(figsize=(15, 10), facecolor='white')

    # Main title with class information
    plt.suptitle(f"Data Augmentation Examples (Class: {class_name})",
                 fontsize=16, fontweight='bold', y=0.98)

    for i, (aug_name, aug_data) in enumerate(aug_types.items()):
        ax = plt.subplot(3, 2, i + 1)
        ax.plot(aug_data.squeeze().numpy(), color=COLORS[i % len(COLORS)], linewidth=2.5)
        ax.set_title(aug_name, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=1.5, length=5)

        if i % 2 == 0:
            ax.set_ylabel("Amplitude", fontsize=12, fontweight='bold')
        if i >= 4:
            ax.set_xlabel("Range Cell", fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def prepare_snr_test_data(dataset, snr_levels=None):
    """准备不同信噪比级别的测试数据"""
    if snr_levels is None:
        snr_levels = Config.noise_robustness['snr_levels']

    # 创建任务生成器
    task_generator = TaskGenerator(dataset, n_way=Config.n_way, k_shot=Config.k_shot, q_query=Config.q_query)

    # 创建增强器
    augmenter = HRRPTransform(augment=False)

    # 为每个信噪比级别生成测试任务
    snr_test_tasks = {}

    for snr in snr_levels:
        tasks = []
        for _ in range(Config.noise_robustness['num_tasks']):
            try:
                support_x, support_y, query_x, query_y = task_generator.generate_task()

                # 添加噪声
                noisy_query_x = []
                for x in query_x:
                    noisy_x = augmenter.add_noise(x.unsqueeze(0), snr).squeeze(0)
                    noisy_query_x.append(noisy_x)

                noisy_query_x = torch.stack(noisy_query_x)
                tasks.append((support_x, support_y, noisy_query_x, query_y))
            except Exception as e:
                print(f"生成SNR={snr}dB任务时出错: {e}")
                continue

        snr_test_tasks[snr] = tasks
        print(f"为SNR={snr}dB生成了 {len(tasks)} 个测试任务")

    return snr_test_tasks