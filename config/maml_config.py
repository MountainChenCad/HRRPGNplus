"""
元学习配置文件 - 定义MAML和小样本学习参数
"""

from config.base_config import BaseConfig


class MAMLConfig(BaseConfig):

    META_EPOCHS = 200  # 元训练总轮次
    # Critical MAML hyperparameters
    INNER_LR = 0.001  # Drastically reduce from current value
    INNER_STEPS = 2  # Reduce to minimum effective steps
    FIRST_ORDER = True  # Use first-order approximation for stability
    META_LR = 0.0003  # Reduce meta learning rate
    META_BATCH_SIZE = 2  # Start with smaller batch size

    # 小样本设置
    N_WAY = 3  # N-way分类
    K_SHOT = 1  # K-shot支持集
    Q_QUERY = 15  # 每类查询样本数

    # 任务采样设置
    TASKS_PER_EPOCH = 200  # 每轮训练采样的任务数
    EVAL_TASKS = 1000  # 验证/测试阶段的任务数

    # 动态图参数
    ALPHA = 0.65  # 距离衰减系数
    DYNAMIC_GRAPH = True  # 是否使用动态图生成
    NUM_HEADS = 4  # 多头注意力数量

    # 课程学习参数
    USE_CURRICULUM = True # 是否使用课程学习
    INIT_TEMP = 1.0  # 初始温度系数
    TEMP_DECAY = 0.99  # 温度衰减率

    # 元学习优化器配置
    META_OPTIMIZER = 'Adam'  # 元优化器类型
    META_MOMENTUM = 0.9  # 元优化器动量

    # 正则化设置
    ORTHO_REG = 0.1  # 正交正则化权重
    METRIC_REG = 0.3  # 度量学习损失权重

    # 模型扩展设置
    USE_META_CONV = True  # 是否使用元卷积
    USE_META_ATTENTION = True  # 是否使用元注意力

    # 快照设置
    SNAPSHOT_INTERVAL = 10  # 每隔多少个epoch保存一次快照
    KEEP_SNAPSHOTS = 5  # 保留的快照数量

    # 验证设置
    VAL_INTERVAL = 10  # 每隔多少个epoch进行一次验证

    @classmethod
    def get_shot_values(cls):
        """返回不同shot设置，用于渐进式实验"""
        return [1, 3, 5, 10]