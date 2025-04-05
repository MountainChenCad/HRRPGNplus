#!/usr/bin/env python
"""
Meta-HRRPNet元学习训练入口
专注于MAML模型的训练流程
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt

from config.maml_config import MAMLConfig
from data.dataset import HRRPDataset
from data.meta_dataset import MetaHRRPDataset, TaskSampler, CurriculumTaskSampler
from models.meta_graph_net import MetaHRRPNet
from trainers.maml_trainer import MAMLTrainer
from utils.visualization import plot_learning_curves, plot_tsne


def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Meta-HRRPNet: MAML Training')

    # 数据路径
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory of datasets (default: data)')

    # GPU设置
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU to use (default: 0)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')

    # 训练设置
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed (default: 3407)')
    parser.add_argument('--meta_epochs', type=int, default=None,
                        help='Number of meta-training epochs (default: use config)')
    parser.add_argument('--meta_batch_size', type=int, default=None,
                        help='Meta batch size (tasks per batch) (default: use config)')
    parser.add_argument('--inner_steps', type=int, default=None,
                        help='Number of inner loop steps (default: use config)')
    parser.add_argument('--inner_lr', type=float, default=None,
                        help='Inner loop learning rate (default: use config)')

    # 元学习设置
    parser.add_argument('--n_way', type=int, default=None,
                        help='N-way classification (default: use config)')
    parser.add_argument('--k_shot', type=int, default=None,
                        help='K-shot learning (default: use config)')
    parser.add_argument('--q_query', type=int, default=None,
                        help='Query samples per class (default: use config)')

    # 模型设置
    parser.add_argument('--dynamic_graph', action='store_true',
                        help='Use dynamic graph generation')
    parser.add_argument('--meta_attention', action='store_true',
                        help='Use meta-attention mechanism')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning')

    # 输出设置
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')

    args = parser.parse_args()
    return args


def setup_logging(log_dir, exp_name):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")

    # 创建日志记录器
    logger = logging.getLogger('meta_train')
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def visualize_learning_progress(history, save_dir):
    """可视化学习进度"""
    # 创建可视化目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘制学习曲线
    plot_learning_curves(
        history,
        metrics=['train_loss', 'train_acc', 'val_loss', 'val_acc'],
        save_path=os.path.join(save_dir, 'learning_curves.png'),
        figsize=(12, 5)
    )

    # 如果使用课程学习，绘制温度参数变化
    if 'temperature' in history and history['temperature'] is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(history['temperature'], 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Temperature')
        plt.title('Curriculum Learning Temperature')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, 'temperature_curve.png'), dpi=300)
        plt.close()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确定实验名称
    if args.exp_name is None:
        args.exp_name = f"meta_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 设置输出目录
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(exp_dir, args.exp_name)
    logger.info(f"开始Meta-HRRPNet元学习训练: {args.exp_name}")
    logger.info(f"命令行参数: {vars(args)}")

    # 创建配置实例
    config = MAMLConfig()

    # 根据命令行参数更新配置
    if args.gpu_id is not None:
        config.GPU_ID = args.gpu_id
    if args.no_cuda:
        config.DEVICE = torch.device('cpu')
    if args.meta_epochs is not None:
        config.META_EPOCHS = args.meta_epochs
    if args.meta_batch_size is not None:
        config.META_BATCH_SIZE = args.meta_batch_size
    if args.inner_steps is not None:
        config.INNER_STEPS = args.inner_steps
    if args.inner_lr is not None:
        config.INNER_LR = args.inner_lr
    if args.n_way is not None:
        config.N_WAY = args.n_way
    if args.k_shot is not None:
        config.K_SHOT = args.k_shot
    if args.q_query is not None:
        config.Q_QUERY = args.q_query

    # 设置模型选项
    config.DYNAMIC_GRAPH = args.dynamic_graph
    config.USE_META_ATTENTION = args.meta_attention
    config.USE_CURRICULUM = args.curriculum

    # 记录配置
    logger.info(f"配置: {config.__class__.__name__}")
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"元学习设置: {config.N_WAY}-way {config.K_SHOT}-shot, {config.Q_QUERY}-query")
    logger.info(
        f"动态图: {config.DYNAMIC_GRAPH}, 元注意力: {config.USE_META_ATTENTION}, 课程学习: {config.USE_CURRICULUM}")

    # 加载数据集
    logger.info("加载数据集...")
    train_dir = os.path.join(args.data_root, 'train_fewshots')
    val_dir = os.path.join(args.data_root, 'val_fewshots')
    test_dir = os.path.join(args.data_root, 'test_fewshots')

    train_dataset = HRRPDataset(train_dir)
    val_dataset = HRRPDataset(val_dir)
    test_dataset = HRRPDataset(test_dir)

    logger.info(
        f"数据集统计 - 训练: {len(train_dataset)}样本, 验证: {len(val_dataset)}样本, 测试: {len(test_dataset)}样本")

    # 创建元学习数据集
    logger.info("创建元学习数据集...")
    meta_train_dataset = MetaHRRPDataset(
        train_dataset,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY,
        num_tasks=config.TASKS_PER_EPOCH,
        task_augment=True
    )

    meta_val_dataset = MetaHRRPDataset(
        val_dataset,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY,
        num_tasks=config.EVAL_TASKS
    )

    meta_test_dataset = MetaHRRPDataset(
        test_dataset,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY,
        num_tasks=config.EVAL_TASKS
    )

    # 创建模型
    logger.info("创建Meta-HRRPNet模型...")
    model = MetaHRRPNet(
        num_classes=config.N_WAY,
        feature_dim=config.FEATURE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        use_dynamic_graph=config.DYNAMIC_GRAPH,
        use_meta_attention=config.USE_META_ATTENTION,
        alpha=config.ALPHA,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT
    )

    # 创建训练器
    logger.info("创建MAML训练器...")
    trainer = MAMLTrainer(
        model,
        meta_train_dataset,
        meta_val_dataset,
        meta_test_dataset,
        config,
        logger
    )

    # 训练模型
    logger.info(f"开始元学习训练, 共{config.META_EPOCHS}轮...")
    history, metrics = trainer.train()

    # 保存训练结果
    save_path = os.path.join(exp_dir, 'meta_train_results.pth')
    torch.save({
        'model': 'Meta-HRRPNet',
        'config': config.get_config_dict(),
        'history': history,
        'metrics': metrics,
        'args': vars(args)
    }, save_path)
    logger.info(f"训练结果已保存至: {save_path}")

    # 打印测试指标
    logger.info("\n" + "=" * 50)
    logger.info("测试结果")
    logger.info("=" * 50)
    logger.info(f"准确率: {metrics['accuracy']:.2f}%")
    logger.info(f"精确率: {metrics['precision']:.2f}%")
    logger.info(f"召回率: {metrics['recall']:.2f}%")
    logger.info(f"F1分数: {metrics['f1']:.2f}%")

    # 每个类别的准确率
    logger.info("\n各类别准确率:")
    for cls, acc in metrics['class_accuracies'].items():
        logger.info(f"类别 {cls}: {acc:.2f}%")

    # 可视化
    if args.visualize:
        logger.info("生成可视化结果...")
        vis_dir = os.path.join(exp_dir, 'visualizations')
        visualize_learning_progress(history, vis_dir)

    logger.info(f"Meta-HRRPNet元学习训练完成: {args.exp_name}")


if __name__ == '__main__':
    main()