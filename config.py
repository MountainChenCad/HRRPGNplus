import os
import torch
import numpy as np
from datetime import datetime


class Config:
    # 基本设置
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集配置
    # data_root = 'datasets/simulated'
    data_root = 'datasets/measured'
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')
    # all_classes = ["F15", "IDF", "F18"]
    all_classes = ['an26', 'citation', 'yar42']
    feature_size = 500  # HRRP特征维度

    # 小样本学习配置
    n_way = 3  # N-way分类
    k_shot = 10  # K-shot (支持集每类样本数)
    q_query = 15  # 查询集每类样本数
    num_tasks = 600  # 测试任务数量

    # 交叉验证方案
    # cross_validation_schemes = [
    #     {
    #         'base_classes': ["F15", "F18", "IDF"],
    #         'novel_classes': ["F15", "F18", "IDF"]
    #     # },
    #     # {
    #     #     'base_classes': ["F15", "F18"],
    #     #     'novel_classes': ["IDF"]
    #     # },
    #     # {
    #     #     'base_classes': ["IDF", "F18"],
    #     #     'novel_classes': ["F15"]
    #     }
    # ]
    cross_validation_schemes = [
        {
            'base_classes': ['an26', 'citation', 'yar42'],
            'novel_classes': ['an26', 'citation', 'yar42']
        # },
        # {
        #     'base_classes': ["F15", "F18"],
        #     'novel_classes': ["IDF"]
        # },
        # {
        #     'base_classes': ["IDF", "F18"],
        #     'novel_classes': ["F15"]
        }
    ]
    current_scheme = 0  # 当前使用的方案索引

    # 模型配置
    hidden_channels = 64
    attention_heads = 4
    graph_conv_layers = 2
    dropout = 0.1

    # 动态图配置
    lambda_mix = 0.3  # 静态和动态图混合比例
    use_dynamic_graph = True  # 是否使用动态图

    # MAML配置
    inner_lr = 0.005  # Reduce from 0.01
    outer_lr = 0.0005  # Reduce from 0.001
    inner_steps = 5  # 内循环更新步数
    task_batch_size = 4  # 每批次任务数
    max_epochs = 300  # 最大迭代轮次
    patience = 100  # 早停耐心值

    # 优化器配置
    weight_decay = 0.01  # L2正则化系数

    # 数据增强配置
    augmentation = False
    noise_levels = [20, 15, 10, 5, 0]  # SNR in dB
    occlusion_ratio = 0.1  # 随机遮挡比例
    phase_jitter = 0.1  # 相位抖动幅度

    # 消融实验配置
    ablation = {
        'dynamic_graph': True,
        'maml': True,
        'lambda_values': [0, 0.25, 0.5, 0.75, 1.0],
        'inner_steps_values': [1, 3, 5, 10],
    }

    # 实验日志配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/experiment_{timestamp}"
    save_dir = f"checkpoints/experiment_{timestamp}"

    @staticmethod
    def create_directories():
        """创建必要的目录"""
        os.makedirs(Config.log_dir, exist_ok=True)
        os.makedirs(Config.save_dir, exist_ok=True)

    @classmethod
    def get_current_scheme(cls):
        """获取当前交叉验证方案"""
        return cls.cross_validation_schemes[cls.current_scheme]

    @staticmethod
    def set_seed(seed=None):
        """设置随机种子"""
        if seed is None:
            seed = Config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    @classmethod
    def update_n_way(cls):
        """根据当前方案更新n_way"""
        scheme = cls.get_current_scheme()
        cls.train_n_way = len(scheme['base_classes'])
        cls.test_n_way = len(scheme['novel_classes'])