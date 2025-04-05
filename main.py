#!/usr/bin/env python
"""
Meta-HRRPNet主执行文件
用于项目入口，提供命令行接口
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import random
from datetime import datetime

from config.base_config import BaseConfig
from config.maml_config import MAMLConfig
from config.exp_config import ExpConfig
from experiments.base_exp import BaseExperiment
from experiments.ablation_exp import AblationExperiment
from experiments.noise_exp import NoiseExperiment


def setup_logging(log_dir, exp_name):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 创建日志记录器
    logger = logging.getLogger('Meta-HRRPNet')
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
    parser = argparse.ArgumentParser(description='Meta-HRRPNet: Meta-Learning for HRRP Target Recognition')

    # 实验类型
    parser.add_argument('--exp_type', type=str, default='base', choices=['base', 'ablation', 'noise', 'all'],
                        help='Experiment type to run (default: base)')

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
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: use config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: use config)')

    # 模型设置
    parser.add_argument('--model', type=str, default='meta', choices=['baseline', 'meta'],
                        help='Model to use (default: meta)')

    # 元学习设置
    parser.add_argument('--n_way', type=int, default=None,
                        help='N-way classification (default: use config)')
    parser.add_argument('--k_shot', type=int, default=None,
                        help='K-shot learning (default: use config)')
    parser.add_argument('--q_query', type=int, default=None,
                        help='Query samples per class (default: use config)')

    # 输出设置
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')

    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: use built-in config)')

    args = parser.parse_args()
    return args


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确定实验名称
    if args.exp_name is None:
        args.exp_name = f"{args.exp_type}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 设置日志
    logger = setup_logging(os.path.join(args.output_dir, 'logs'), args.exp_name)
    logger.info(f"开始Meta-HRRPNet实验: {args.exp_name}")
    logger.info(f"命令行参数: {vars(args)}")

    # 选择配置类
    if args.exp_type == 'base':
        config_class = MAMLConfig if args.model == 'meta' else BaseConfig
    else:
        config_class = ExpConfig

    # 创建配置实例
    config = config_class()

    # 根据命令行参数更新配置
    if args.gpu_id is not None:
        config.GPU_ID = args.gpu_id
    if args.no_cuda:
        config.DEVICE = torch.device('cpu')
    if args.epochs is not None:
        if args.model == 'meta':
            config.META_EPOCHS = args.epochs
        else:
            config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        if args.model == 'meta':
            config.META_BATCH_SIZE = args.batch_size
        else:
            config.BATCH_SIZE = args.batch_size
    if args.n_way is not None:
        config.N_WAY = args.n_way
    if args.k_shot is not None:
        config.K_SHOT = args.k_shot
    if args.q_query is not None:
        config.Q_QUERY = args.q_query

    # 显示当前设置
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"配置: {config.__class__.__name__}")
    if args.model == 'meta':
        logger.info(f"元学习设置: {config.N_WAY}-way {config.K_SHOT}-shot")

    # 运行实验
    if args.exp_type == 'base':
        logger.info("运行基础实验...")
        exp = BaseExperiment(config, args.data_root, args.output_dir, args.seed)
        if args.model == 'baseline':
            results = exp.run_baseline()
        elif args.model == 'meta':
            results = exp.run_meta_model()
        else:
            results = exp.run_comparison()

    elif args.exp_type == 'ablation':
        logger.info("运行消融实验...")
        exp = AblationExperiment(config, args.data_root, args.output_dir, args.seed)
        results = exp.run_ablation()

    elif args.exp_type == 'noise':
        logger.info("运行噪声鲁棒性实验...")
        exp = NoiseExperiment(config, args.data_root, args.output_dir, args.seed)
        results = exp.run_noise_experiment()

    elif args.exp_type == 'all':
        logger.info("运行所有实验...")

        # 基础实验
        logger.info("\n" + "=" * 50)
        logger.info("1. 基础实验")
        logger.info("=" * 50)
        base_exp = BaseExperiment(config, args.data_root, args.output_dir, args.seed)
        base_results = base_exp.run_comparison()

        # 消融实验
        logger.info("\n" + "=" * 50)
        logger.info("2. 消融实验")
        logger.info("=" * 50)
        ablation_exp = AblationExperiment(config, args.data_root, args.output_dir, args.seed)
        ablation_results = ablation_exp.run_ablation()

        # 噪声实验
        logger.info("\n" + "=" * 50)
        logger.info("3. 噪声鲁棒性实验")
        logger.info("=" * 50)
        noise_exp = NoiseExperiment(config, args.data_root, args.output_dir, args.seed)
        noise_results = noise_exp.run_noise_experiment()

        # 汇总结果
        results = {
            'base': base_results,
            'ablation': ablation_results,
            'noise': noise_results
        }

        # 保存所有结果
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_results.pth')
        torch.save(results, save_path)
        logger.info(f"所有实验结果已保存至: {save_path}")

    logger.info(f"Meta-HRRPNet实验完成: {args.exp_name}")


if __name__ == '__main__':
    main()