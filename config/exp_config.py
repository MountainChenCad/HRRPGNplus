"""
实验配置文件 - 定义不同实验场景的参数
"""

import numpy as np
from config.maml_config import MAMLConfig


class ExpConfig(MAMLConfig):
    # 实验类型
    EXP_TYPE = 'full'  # 可选: 'baseline', 'ablation', 'noise', 'full'

    # 消融实验组合
    ABLATION_GROUPS = {
        'base': {
            'DYNAMIC_GRAPH': False,
            'USE_META_CONV': False,
            'USE_CURRICULUM': False
        },
        'dyn_graph': {
            'DYNAMIC_GRAPH': True,
            'USE_META_CONV': False,
            'USE_CURRICULUM': False
        },
        'meta_conv': {
            'DYNAMIC_GRAPH': True,
            'USE_META_CONV': True,
            'USE_CURRICULUM': False
        },
        'curriculum': {
            'DYNAMIC_GRAPH': True,
            'USE_META_CONV': True,
            'USE_CURRICULUM': True
        },
        'full': {
            'DYNAMIC_GRAPH': True,
            'USE_META_CONV': True,
            'USE_CURRICULUM': True,
            'USE_META_ATTENTION': True
        }
    }

    # 模型对比列表
    COMPARE_MODELS = [
        'SVM', 'CNN-1D', 'GCN', 'GAT',
        'MatchingNet', 'ProtoNet', 'HRRPGraphNet', 'Meta-HRRPNet'
    ]

    # 噪声实验设置
    NOISE_TYPES = ['gaussian', 'impulse', 'speckle']
    SNR_LEVELS = [-5, 0, 5, 10, 15, 20]  # 信噪比范围，单位dB

    # 噪声添加函数配置
    NOISE_PARAMS = {
        'gaussian': {'mean': 0, 'scale_range': [0.01, 0.5]},
        'impulse': {'prob_range': [0.01, 0.3], 'strength_range': [0.5, 2.0]},
        'speckle': {'scale_range': [0.01, 0.5]}
    }

    # 交叉验证设置
    CROSS_VAL_FOLDS = 5

    # 评估指标设置
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']

    # 可视化设置
    VIS_TSNE = True  # 是否绘制t-SNE图
    VIS_CONFUSION = True  # 是否绘制混淆矩阵
    VIS_ATTENTION = True  # 是否可视化注意力权重
    VIS_GRAPH = True  # 是否可视化图结构
    VIS_CURVE = True  # 是否绘制学习曲线

    # 统计显著性测试设置
    SIG_TEST = True  # 是否进行显著性测试
    ALPHA_VALUE = 0.05  # 显著性水平

    # 特殊实验组合
    FEW_SHOT_RANGE = [(5, k) for k in [1, 3, 5, 10]]  # (n_way, k_shot)组合

    # 跨数据集实验
    CROSS_DATASET = {
        'source': 'simulation',  # 源数据集（仿真数据）
        'target': 'MSTAR',  # 目标数据集（真实数据）
        'modes': ['fine_tuning', 'frozen_extractor']  # 迁移方式
    }

    @classmethod
    def get_noise_params(cls, noise_type, snr):
        """根据噪声类型和信噪比返回噪声参数"""
        base_params = cls.NOISE_PARAMS[noise_type].copy()
        # 根据SNR调整噪声参数
        if 'scale_range' in base_params:
            idx = np.clip(cls.SNR_LEVELS.index(snr), 0, len(cls.SNR_LEVELS) - 1)
            scale = np.interp(idx, [0, len(cls.SNR_LEVELS) - 1], base_params['scale_range'])
            base_params['scale'] = scale
            del base_params['scale_range']
        if 'prob_range' in base_params:
            idx = np.clip(cls.SNR_LEVELS.index(snr), 0, len(cls.SNR_LEVELS) - 1)
            prob = np.interp(idx, [0, len(cls.SNR_LEVELS) - 1], base_params['prob_range'])
            base_params['prob'] = prob
            del base_params['prob_range']
        if 'strength_range' in base_params:
            idx = np.clip(cls.SNR_LEVELS.index(snr), 0, len(cls.SNR_LEVELS) - 1)
            strength = np.interp(idx, [0, len(cls.SNR_LEVELS) - 1], base_params['strength_range'])
            base_params['strength'] = strength
            del base_params['strength_range']
        return base_params

    @classmethod
    def get_ablation_config(cls, group):
        """获取指定消融组的配置参数"""
        if group not in cls.ABLATION_GROUPS:
            raise ValueError(f"Unknown ablation group: {group}")
        return cls.ABLATION_GROUPS[group]