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
    k_shot = 5  # K-shot (支持集每类样本数)
    q_query = 15  # 查询集每类样本数
    num_tasks = 100  # 600

    # 交叉验证方案
    cross_validation_schemes = [
        {
            'base_classes': ['an26', 'citation', 'yar42'],
            'novel_classes': ['an26', 'citation', 'yar42']
        }
    ]
    current_scheme = 0  # 当前使用的方案索引

    # 模型配置
    hidden_channels = 64
    attention_heads = 4
    graph_conv_layers = 2 # 3
    dropout = 0.1

    # 动态图配置
    lambda_mix = 0.3  # 静态和动态图混合比例
    use_dynamic_graph = True  # 是否使用动态图

    # MAML++配置
    inner_lr = 0.005  # 初始内循环学习率基准值
    outer_lr = 0.0005  # 初始外循环学习率
    inner_steps = 5  # 内循环更新步数
    task_batch_size = 1  # 4
    max_epochs = 300  # 最大迭代轮次
    patience = 20  # 100

    # MAML++ 多步骤损失权重
    # 按步骤递增的权重，越靠后的步骤权重越大
    step_loss_weights = [0.1, 0.2, 0.3, 0.4, 1.0]

    # 学习率配置
    use_multi_step_loss = True  # 是否使用多步骤损失
    use_per_step_lr = True  # 是否使用每步自适应学习率
    use_per_layer_lr = True  # 是否使用每层自适应学习率

    # 内循环学习率初始值
    # 每层每步的学习率初始值
    # 格式: {'layer_name': [step1_lr, step2_lr, ...]}
    layer_lr_init = {
        'encoder.feature_extractor': [0.004, 0.004, 0.003, 0.003, 0.002],
        'encoder.graph_generator': [0.005, 0.005, 0.004, 0.004, 0.003],
        'encoder.graph_convs': [0.006, 0.006, 0.005, 0.005, 0.004],
        'encoder.pooling': [0.004, 0.004, 0.003, 0.003, 0.002],
        'encoder.fc': [0.003, 0.003, 0.002, 0.002, 0.001]
    }

    # 学习率退火配置
    use_lr_annealing = True  # 是否使用学习率退火
    min_outer_lr = 0.00001  # 外循环学习率最小值
    T_max = 200  # 余弦退火周期

    # 梯度阶数退火配置
    use_second_order = True  # 是否使用二阶梯度
    second_order_start_epoch = 50  # 开始使用二阶梯度的轮次

    # BN层处理配置
    per_step_bn_statistics = True  # 每步更新BN统计量

    # 优化器配置
    weight_decay = 0.01  # L2正则化系数

    # 数据增强配置
    augmentation = False  # 是否使用数据增强
    noise_levels = [20, 15, 10, 5, 0]  # SNR in dB
    occlusion_ratio = 0.1  # 随机遮挡比例
    phase_jitter = 0.1  # 相位抖动幅度

    # ==================== 实验配置 ====================

    # 实验评估配置
    evaluation_shots = [1]  # [1, 5, 10, 20]
    evaluation_ways = [3]  # N-way K-shot设置的N值
    evaluation_metrics = ['accuracy', 'f1_score', 'confusion_matrix']  # 评估指标

    # 噪声鲁棒性实验配置
    noise_robustness = {
        'enabled': True,
        'snr_levels': [20, 15, 10, 5, 0, -5],  # 信噪比水平 (dB)
        'num_tasks': 100  # 100
    }

    # 数据稀疏性实验配置
    data_sparsity = {
        'enabled': True,
        'shot_levels': [1, 5, 10, 20],  # K-shot设置
        'num_tasks': 100  # 100
    }

    # ========== 消融实验配置 ==========

    # 基本消融实验配置
    ablation = {
        'dynamic_graph': True,  # 是否测试动态图模块
        'maml': True,  # 是否测试元学习模块
        'lambda_values': [0, 0.25, 0.5, 0.75, 1.0],  # 混合系数γ的值
        'inner_steps_values': [1, 3, 5, 10],  # 内循环步数的值
    }

    # 图结构建模分析配置
    graph_structure_ablation = {
        'enabled': True,
        'types': ['static', 'dynamic', 'hybrid'],  # 图结构类型
        'lambda_values': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # 混合系数γ的值
        'visualization_enabled': True  # 是否启用邻接矩阵可视化
    }

    # GNN架构组件分析配置
    gnn_architecture_ablation = {
        'enabled': True,
        'graph_conv_layers_values': [1, 2, 3, 4],  # 图卷积层数的值
        'attention_heads_values': [2, 4, 8],  # 注意力头数的值
        'pooling_strategies': ['attention', 'mean', 'max']  # 池化策略
    }

    # 元学习优化分析配置
    meta_learning_ablation = {
        'enabled': True,
        'per_layer_lr_enabled': [True, False],  # 是否使用每层学习率
        'per_step_lr_enabled': [True, False],  # 是否使用每步学习率
        'multi_step_loss_enabled': [True, False],  # 是否使用多步损失
        'second_order_enabled': [True, False]  # 是否使用二阶导数
    }

    # ========== 基线模型比较配置 ==========

    # 传统雷达目标识别方法
    traditional_baselines = {
        'enabled': True,
        'methods': ['PCA+SVM', 'Template Matching']
    }

    # 深度学习方法
    dl_baselines = {
        'enabled': True,
        'methods': ['CNN', 'LSTM', 'GCN', 'GAT']
    }

    # 少样本学习方法
    fsl_baselines = {
        'enabled': True,
        'methods': ['ProtoNet', 'MatchingNet']
    }

    # ========== 可视化与解释性分析配置 ==========

    # 注意力机制可视化配置
    attention_visualization = {
        'enabled': True,
        'num_samples': 5,  # 可视化样本数
        'save_dir': 'visualizations/attention'  # 保存目录
    }

    # 动态图结构分析配置
    dynamic_graph_visualization = {
        'enabled': True,
        'num_samples': 5,  # 可视化样本数
        'save_dir': 'visualizations/dynamic_graph'  # 保存目录
    }

    # 特征空间可视化配置
    feature_visualization = {
        'enabled': True,
        'method': 't-SNE',  # 降维方法
        'perplexity': 30,  # t-SNE参数
        'save_dir': 'visualizations/features'  # 保存目录
    }

    # ========== 计算复杂度分析配置 ==========

    computational_complexity = {
        'enabled': True,
        'measure_inference_time': True,  # 是否测量推理时间
        'measure_memory_usage': True,  # 是否测量内存使用
        'measure_flops': True,  # 是否测量浮点运算数
        'num_runs': 100  # 推理时间测量的运行次数
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

        # 创建可视化目录
        if Config.attention_visualization['enabled']:
            os.makedirs(os.path.join(Config.log_dir, Config.attention_visualization['save_dir']), exist_ok=True)

        if Config.dynamic_graph_visualization['enabled']:
            os.makedirs(os.path.join(Config.log_dir, Config.dynamic_graph_visualization['save_dir']), exist_ok=True)

        if Config.feature_visualization['enabled']:
            os.makedirs(os.path.join(Config.log_dir, Config.feature_visualization['save_dir']), exist_ok=True)

    @classmethod
    def load_experiment(cls, exp_id):
        """Load configuration from a previous experiment"""
        log_dir = f"logs/experiment_{exp_id}"
        save_dir = f"checkpoints/experiment_{exp_id}"

        if not os.path.exists(log_dir):
            print(f"Error: Experiment directory not found: {log_dir}")
            return False

        # Update directories
        cls.timestamp = exp_id
        cls.log_dir = log_dir
        cls.save_dir = save_dir
        print(f"Loaded experiment {exp_id} configuration")
        return True

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

    @classmethod
    def get_step_lr(cls, layer_name, step_idx):
        """获取指定层和步骤的学习率"""
        if not cls.use_per_step_lr or not cls.use_per_layer_lr:
            return cls.inner_lr

        # 尝试获取指定层的学习率配置
        if layer_name in cls.layer_lr_init:
            step_lrs = cls.layer_lr_init[layer_name]
            if step_idx < len(step_lrs):
                return step_lrs[step_idx]

        # 若没有找到匹配的配置，返回默认学习率
        return cls.inner_lr

    @classmethod
    def get_loss_weight(cls, step_idx):
        """获取指定步骤的损失权重"""
        if not cls.use_multi_step_loss:
            # 只有最后一步有权重
            return 1.0 if step_idx == cls.inner_steps - 1 else 0.0

        # 使用预配置的权重
        if step_idx < len(cls.step_loss_weights):
            return cls.step_loss_weights[step_idx]

        # 默认权重
        return 1.0 if step_idx == cls.inner_steps - 1 else 0.2