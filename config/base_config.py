"""
基础配置文件 - 定义模型和训练的基本参数
"""

import os
import torch
from datetime import datetime


class BaseConfig:
    # 路径设置
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    SAVE_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')
    RESULT_DIR = os.path.join(ROOT_DIR, 'results')

    # 创建必要的目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 实验ID，用于区分不同实验
    EXP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 数据集设置
    RANDOM_SEED = 3407  # 固定随机种子，保证实验可重复
    SHUFFLE = True
    NUM_WORKERS = 8

    # GPU设置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID = 0  # 指定使用的GPU ID

    # 标准训练设置
    BATCH_SIZE = 128
    NUM_EPOCHS = 300
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.01

    # 模型架构设置
    FEATURE_DIM = 500  # HRRP特征维度
    HIDDEN_DIM = 32  # 隐藏层维度
    NUM_CLASSES = 3  # 目标类别数（默认3类，可根据数据集调整）
    DROPOUT = 0.1  # Dropout率

    # 早停设置
    PATIENCE = 20  # 早停耐心值
    DELTA = 0.001  # 早停判定阈值

    # 保存和加载设置
    SAVE_FREQ = 10  # 每隔多少个epoch保存一次模型
    SAVE_BEST = True  # 是否只保存最佳模型

    # 损失函数权重
    LAMBDA_L2 = 0.01  # L2正则化权重

    @classmethod
    def get_config_dict(cls):
        """返回配置字典，便于日志记录"""
        config_dict = {}
        for key in dir(cls):
            if not key.startswith('__') and not callable(getattr(cls, key)):
                config_dict[key] = getattr(cls, key)
        return config_dict