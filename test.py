#!/usr/bin/env python
"""
Meta-HRRPNet测试与评估脚本
用于评估模型在不同条件下的性能
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
from tqdm import tqdm

from config.base_config import BaseConfig
from config.maml_config import MAMLConfig
from data.dataset import HRRPDataset
from data.meta_dataset import MetaHRRPDataset, TaskSampler
from models.baseline_gcn import HRRPGraphNet
from models.meta_graph_net import MetaHRRPNet
from utils.metrics import calculate_metrics, meta_learning_metrics
from utils.visualization import (
    plot_confusion_matrix, plot_tsne, plot_shot_comparison,
    plot_attention_weights, plot_graph, plot_noise_robustness
)


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
    parser = argparse.ArgumentParser(description='Meta-HRRPNet: Model Testing')

    # 模型设置
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'meta'],
                        help='Model type to test (baseline or meta)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')

    # 数据路径
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory of datasets (default: data)')
    parser.add_argument('--test_dir', type=str, default='test',
                        help='Test directory within data_root (default: test)')

    # GPU设置
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU to use (default: 0)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    # 测试设置
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed (default: 3407)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for baseline testing (default: 128)')

    # 元学习设置
    parser.add_argument('--n_way', type=int, default=5,
                        help='N-way classification for meta-testing (default: 5)')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='K-shot learning for meta-testing (default: 1)')
    parser.add_argument('--num_tasks', type=int, default=1000,
                        help='Number of tasks to test (default: 1000)')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='Number of adaptation steps (default: 5)')
    parser.add_argument('--inner_lr', type=float, default=0.05,
                        help='Inner loop learning rate (default: 0.05)')

    # 测试类型
    parser.add_argument('--test_type', type=str, default='standard',
                        choices=['standard', 'shot', 'noise', 'all'],
                        help='Type of test to perform (default: standard)')

    # 噪声测试设置
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'impulse', 'speckle', 'all'],
                        help='Type of noise for noise testing (default: gaussian)')
    parser.add_argument('--snr_range', type=str, default='-5,0,5,10,15,20',
                        help='SNR range in dB, comma-separated (default: -5,0,5,10,15,20)')

    # shot测试设置
    parser.add_argument('--shot_range', type=str, default='1,3,5,10',
                        help='K-shot range, comma-separated (default: 1,3,5,10)')

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
    logger = logging.getLogger('test')
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


def generate_distance_matrix(N, alpha=1.0):
    """生成距离矩阵"""
    distance_matrix = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = 1 / (alpha * abs(i - j) + 1)
    return distance_matrix


def test_baseline_model(model, test_dataset, distance_matrix, batch_size, device, logger):
    """测试标准模型"""
    logger.info("评估标准模型...")

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    # 收集预测结果
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)

            # 前向传播
            if hasattr(model, 'use_dynamic_graph') and model.use_dynamic_graph:
                logits, _ = model(data, distance_matrix)
            else:
                logits = model(data, distance_matrix)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算性能指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    # 打印结果
    logger.info(f"准确率: {metrics['accuracy']:.2f}%")
    logger.info(f"精确率: {metrics['precision']:.2f}%")
    logger.info(f"召回率: {metrics['recall']:.2f}%")
    logger.info(f"F1分数: {metrics['f1']:.2f}%")

    # 每个类别的准确率
    logger.info("\n各类别准确率:")
    for cls, acc in metrics['class_accuracies'].items():
        logger.info(f"类别 {cls}: {acc:.2f}%")

    return metrics, all_preds, all_labels, all_probs


def test_meta_model(model, test_dataset, distance_matrix, args, device, logger):
    """测试元学习模型"""
    logger.info(f"评估元学习模型 ({args.n_way}-way {args.k_shot}-shot)...")

    # 创建元学习测试数据集
    meta_test_dataset = MetaHRRPDataset(
        test_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=15,  # 查询集较大，确保评估可靠性
        num_tasks=args.num_tasks
    )

    # 创建任务采样器
    task_sampler = TaskSampler(
        test_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=15,
        num_tasks=args.num_tasks,
        fixed_tasks=True
    )

    model.eval()
    all_task_accuracies = []
    all_preds = []
    all_labels = []

    # 对每个任务进行评估
    for task_idx in tqdm(range(args.num_tasks), desc="Testing"):
        # 采样任务
        task = task_sampler.tasks[task_idx]

        support_x = task['support_x'].to(device)
        support_y = task['support_y'].to(device)
        query_x = task['query_x'].to(device)
        query_y = task['query_y'].to(device)

        # 模型适应（内循环）
        adapted_state_dict = adapt_model(model, support_x, support_y,
                                         distance_matrix, args.inner_steps,
                                         args.inner_lr, device)

        # 在查询集上评估
        with torch.no_grad():
            if hasattr(model, 'use_dynamic_graph') and model.use_dynamic_graph:
                logits, _ = model(query_x, distance_matrix)
            else:
                logits = model(query_x, distance_matrix)

            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == query_y).float().mean().item() * 100
            all_task_accuracies.append(accuracy)

            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_y.cpu().numpy())

        # 恢复原始模型状态
        model.load_state_dict(adapted_state_dict)

    # 计算元学习指标
    meta_metrics = meta_learning_metrics(all_task_accuracies)

    # 计算整体指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_metrics = calculate_metrics(all_labels, all_preds)

    # 合并指标
    metrics = {**meta_metrics, **overall_metrics}

    # 打印结果
    logger.info(f"平均准确率: {meta_metrics['mean_accuracy']:.2f}% ± {meta_metrics['confidence_interval']:.2f}%")
    logger.info(f"最小准确率: {meta_metrics['min_accuracy']:.2f}%")
    logger.info(f"最大准确率: {meta_metrics['max_accuracy']:.2f}%")
    logger.info(f"标准差: {meta_metrics['std_accuracy']:.2f}%")
    logger.info(f"成功率 (>90%): {meta_metrics['success_rate']:.2f}%")

    # 每个类别的准确率
    logger.info("\n各类别准确率:")
    for cls, acc in overall_metrics['class_accuracies'].items():
        logger.info(f"类别 {cls}: {acc:.2f}%")

    return metrics, all_preds, all_labels


def adapt_model(model, support_x, support_y, distance_matrix, inner_steps, inner_lr, device):
    """模型适应（内循环）"""
    # 保存原始状态
    original_state = {name: param.clone() for name, param in model.named_parameters()}

    # 内循环适应
    for _ in range(inner_steps):
        # 前向传播
        if hasattr(model, 'use_dynamic_graph') and model.use_dynamic_graph:
            logits, _ = model(support_x, distance_matrix)
        else:
            logits = model(support_x, distance_matrix)

        # 计算损失
        loss = torch.nn.functional.cross_entropy(logits, support_y)

        # 计算梯度
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)

        # 更新权重
        for param, grad in zip(model.parameters(), grads):
            param.data = param.data - inner_lr * grad

    # 创建适应后的状态字典
    adapted_state_dict = model.state_dict()

    # 恢复原始状态
    for name, param in model.named_parameters():
        param.data = original_state[name]

    return adapted_state_dict


def test_noise_robustness(model, test_dataset, distance_matrix, args, device, logger):
    """测试噪声鲁棒性"""
    logger.info("评估噪声鲁棒性...")

    # 解析SNR范围
    snr_levels = [int(snr) for snr in args.snr_range.split(',')]

    # 确定噪声类型
    noise_types = ['gaussian', 'impulse', 'speckle'] if args.noise_type == 'all' else [args.noise_type]

    # 存储结果
    results = {noise_type: {} for noise_type in noise_types}

    # 在干净数据上测试基准性能
    if args.model_type == 'baseline':
        clean_metrics, _, _, _ = test_baseline_model(
            model, test_dataset, distance_matrix, args.batch_size, device, logger
        )
        clean_accuracy = clean_metrics['accuracy']
    else:
        clean_metrics, _, _ = test_meta_model(
            model, test_dataset, distance_matrix, args, device, logger
        )
        clean_accuracy = clean_metrics['mean_accuracy']

    logger.info(f"干净数据准确率: {clean_accuracy:.2f}%")

    # 对每种噪声类型和信噪比进行测试
    for noise_type in noise_types:
        logger.info(f"\n测试 {noise_type} 噪声...")
        results[noise_type]['clean'] = clean_accuracy

        for snr in snr_levels:
            logger.info(f"SNR = {snr}dB")

            # 创建噪声数据集
            noisy_dataset = create_noisy_dataset(test_dataset, noise_type, snr)

            # 测试性能
            if args.model_type == 'baseline':
                metrics, _, _, _ = test_baseline_model(
                    model, noisy_dataset, distance_matrix, args.batch_size, device, logger
                )
                accuracy = metrics['accuracy']
            else:
                metrics, _, _ = test_meta_model(
                    model, noisy_dataset, distance_matrix, args, device, logger
                )
                accuracy = metrics['mean_accuracy']

            # 计算鲁棒性比例
            robustness = (accuracy / clean_accuracy) * 100

            logger.info(f"准确率: {accuracy:.2f}% (鲁棒性: {robustness:.2f}%)")

            # 存储结果
            results[noise_type][snr] = {
                'accuracy': accuracy,
                'robustness': robustness,
                'metrics': metrics
            }

    return results, clean_accuracy


def create_noisy_dataset(dataset, noise_type, snr):
    """创建带噪声的数据集"""
    # 计算噪声参数
    if noise_type == 'gaussian':
        noise_params = {'scale': 10 ** (-(snr / 20))}
    elif noise_type == 'impulse':
        noise_params = {'prob': max(0.01, min(0.3, 0.3 * 10 ** (-(snr / 20)))),
                        'strength': 0.5 + (20 - snr) / 10}
    elif noise_type == 'speckle':
        noise_params = {'scale': 10 ** (-(snr / 20))}
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")

    # 创建噪声数据集
    noisy_dataset = dataset.add_noise(noise_type, **noise_params)

    return noisy_dataset


def test_different_shots(model, test_dataset, distance_matrix, args, device, logger):
    """测试不同shot设置下的性能"""
    logger.info("评估不同shot设置下的性能...")

    # 解析shot范围
    shot_values = [int(k) for k in args.shot_range.split(',')]

    # 存储结果
    results = {}

    # 对每个shot设置进行测试
    for k_shot in shot_values:
        logger.info(f"\n测试 {k_shot}-shot 设置...")

        # 更新参数
        args.k_shot = k_shot

        # 测试性能
        if args.model_type == 'baseline':
            # 为基线模型限制每类样本数
            limited_dataset = create_limited_dataset(test_dataset, k_shot)
            metrics, _, _, _ = test_baseline_model(
                model, limited_dataset, distance_matrix, args.batch_size, device, logger
            )
        else:
            metrics, _, _ = test_meta_model(
                model, test_dataset, distance_matrix, args, device, logger
            )

        # 存储结果
        results[k_shot] = metrics

    return results, shot_values


def create_limited_dataset(dataset, k_shot):
    """创建限制每类样本数的数据集"""
    # 这里简化实现，实际上应该随机选择k_shot个样本
    # 在真实应用中应更加细致地实现采样逻辑

    limited_dataset = HRRPDataset(dataset.root_dir)

    # 修改采样逻辑，限制每类样本数
    limited_dataset.samples_by_class = {
        cls: samples[:k_shot] if len(samples) > k_shot else samples
        for cls, samples in dataset.samples_by_class.items()
    }

    # 重建文件列表
    limited_dataset.file_list = []
    for samples in limited_dataset.samples_by_class.values():
        limited_dataset.file_list.extend([os.path.basename(s) for s in samples])

    return limited_dataset


def visualize_results(model, test_dataset, metrics, predictions, labels, args, exp_dir, logger):
    """可视化结果"""
    logger.info("生成可视化结果...")

    # 创建可视化目录
    vis_dir = os.path.join(exp_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 绘制混淆矩阵
    if 'confusion_matrix' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=[str(i) for i in range(len(metrics['class_accuracies']))],
            title='Confusion Matrix',
            save_path=os.path.join(vis_dir, 'confusion_matrix.png')
        )

    # 特征可视化（需要额外运行模型以提取特征）
    if hasattr(model, 'get_embedding'):
        logger.info("生成t-SNE特征可视化...")

        # 采样一部分数据进行可视化
        vis_samples = min(1000, len(test_dataset))
        indices = np.random.choice(len(test_dataset), vis_samples, replace=False)

        features = []
        feature_labels = []

        # 提取特征
        model.eval()
        with torch.no_grad():
            for idx in tqdm(indices, desc="Extracting features"):
                data, label = test_dataset[idx]
                data = data.unsqueeze(0).to(args.device)

                if hasattr(model, 'get_embedding'):
                    feature = model.get_embedding(data)
                    # 将特征压缩为向量
                    feature = feature.mean(dim=1).cpu().numpy()
                    features.append(feature)
                    feature_labels.append(label.item())

        # 绘制t-SNE
        if features:
            features = np.vstack(features)
            feature_labels = np.array(feature_labels)

            plot_tsne(
                features, feature_labels,
                title='Feature Embedding Visualization',
                save_path=os.path.join(vis_dir, 'tsne_visualization.png')
            )

    # 如果是噪声测试，绘制噪声鲁棒性图
    if args.test_type == 'noise' and hasattr(args, 'noise_results'):
        logger.info("生成噪声鲁棒性可视化...")

        noise_types = ['gaussian', 'impulse', 'speckle'] if args.noise_type == 'all' else [args.noise_type]
        snr_levels = [int(snr) for snr in args.snr_range.split(',')]

        for noise_type in noise_types:
            # 提取性能数据
            accuracies = [args.noise_results[noise_type][snr]['accuracy'] for snr in snr_levels]
            robustness = [args.noise_results[noise_type][snr]['robustness'] for snr in snr_levels]

            # 绘制性能曲线
            plt.figure(figsize=(10, 6))
            plt.plot(snr_levels, accuracies, 'o-')
            plt.axhline(y=args.noise_results[noise_type]['clean'], color='r', linestyle='--',
                        label=f'Clean: {args.noise_results[noise_type]["clean"]:.1f}%')

            plt.xlabel('SNR (dB)')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Performance under {noise_type.capitalize()} Noise')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plt.savefig(os.path.join(vis_dir, f'{noise_type}_performance.png'), dpi=300)
            plt.close()

            # 绘制鲁棒性曲线
            plt.figure(figsize=(10, 6))
            plt.plot(snr_levels, robustness, 's-')
            plt.axhline(y=100, color='gray', linestyle='--', label='No degradation')

            plt.xlabel('SNR (dB)')
            plt.ylabel('Robustness (%)')
            plt.title(f'Robustness under {noise_type.capitalize()} Noise')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plt.savefig(os.path.join(vis_dir, f'{noise_type}_robustness.png'), dpi=300)
            plt.close()

    # 如果是shot测试，绘制shot比较图
    if args.test_type == 'shot' and hasattr(args, 'shot_results'):
        logger.info("生成shot比较可视化...")

        shot_values = [int(k) for k in args.shot_range.split(',')]
        accuracies = []

        for k in shot_values:
            if args.model_type == 'baseline':
                accuracies.append(args.shot_results[k]['accuracy'])
            else:
                accuracies.append(args.shot_results[k]['mean_accuracy'])

        plt.figure(figsize=(10, 6))
        plt.plot(shot_values, accuracies, 'o-')

        plt.xlabel('Samples per Class (K-shot)')
        plt.ylabel('Accuracy (%)')
        plt.title('Performance vs. Number of Shots')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(os.path.join(vis_dir, 'shot_comparison.png'), dpi=300)
        plt.close()

    logger.info(f"可视化结果已保存至: {vis_dir}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确定实验名称
    if args.exp_name is None:
        args.exp_name = f"test_{args.model_type}_{args.test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 设置输出目录
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(exp_dir, args.exp_name)
    logger.info(f"开始Meta-HRRPNet测试: {args.exp_name}")
    logger.info(f"命令行参数: {vars(args)}")

    # 设置设备
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info("加载测试数据集...")
    test_dir = os.path.join(args.data_root, args.test_dir)
    test_dataset = HRRPDataset(test_dir)
    logger.info(f"测试数据集: {len(test_dataset)}样本, {len(test_dataset.classes)}类")

    # 加载模型
    logger.info(f"加载{args.model_type}模型...")

    if args.model_type == 'baseline':
        # 加载标准模型
        model = HRRPGraphNet(num_classes=len(test_dataset.classes))
        model.to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # 加载元学习模型
        model = MetaHRRPNet(
            num_classes=args.n_way,
            use_dynamic_graph=True,  # 假设使用动态图
            use_meta_attention=True  # 假设使用元注意力
        )
        model.to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # 生成距离矩阵
    feature_dim = 500  # 假设HRRP特征长度为500
    distance_matrix = generate_distance_matrix(feature_dim).to(device)

    # 根据测试类型执行测试
    if args.test_type == 'standard':
        # 标准测试
        if args.model_type == 'baseline':
            metrics, predictions, labels, probs = test_baseline_model(
                model, test_dataset, distance_matrix, args.batch_size, device, logger
            )
        else:
            metrics, predictions, labels = test_meta_model(
                model, test_dataset, distance_matrix, args, device, logger
            )
            probs = None

        # 保存结果
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels
        }
        if probs is not None:
            results['probabilities'] = probs

        save_path = os.path.join(exp_dir, 'test_results.pth')
        torch.save(results, save_path)
        logger.info(f"测试结果已保存至: {save_path}")

        # 可视化结果
        if args.visualize:
            visualize_results(model, test_dataset, metrics, predictions, labels, args, exp_dir, logger)

    elif args.test_type == 'noise':
        # 噪声鲁棒性测试
        noise_results, clean_accuracy = test_noise_robustness(
            model, test_dataset, distance_matrix, args, device, logger
        )

        # 保存结果
        args.noise_results = noise_results
        save_path = os.path.join(exp_dir, 'noise_test_results.pth')
        torch.save({
            'noise_results': noise_results,
            'clean_accuracy': clean_accuracy,
            'args': vars(args)
        }, save_path)
        logger.info(f"噪声测试结果已保存至: {save_path}")

        # 可视化结果
        if args.visualize:
            logger.info("生成噪声测试可视化...")
            vis_dir = os.path.join(exp_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 调用可视化功能
            visualize_results(model, test_dataset, {}, None, None, args, exp_dir, logger)

    elif args.test_type == 'shot':
        # 不同shot设置测试
        shot_results, shot_values = test_different_shots(
            model, test_dataset, distance_matrix, args, device, logger
        )

        # 保存结果
        args.shot_results = shot_results
        save_path = os.path.join(exp_dir, 'shot_test_results.pth')
        torch.save({
            'shot_results': shot_results,
            'shot_values': shot_values,
            'args': vars(args)
        }, save_path)
        logger.info(f"Shot测试结果已保存至: {save_path}")

        # 可视化结果
        if args.visualize:
            logger.info("生成shot测试可视化...")
            vis_dir = os.path.join(exp_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 调用可视化功能
            visualize_results(model, test_dataset, {}, None, None, args, exp_dir, logger)

    elif args.test_type == 'all':
        logger.info("执行所有测试类型...")

        # 标准测试
        logger.info("\n" + "=" * 50)
        logger.info("1. 标准测试")
        logger.info("=" * 50)
        if args.model_type == 'baseline':
            standard_metrics, predictions, labels, probs = test_baseline_model(
                model, test_dataset, distance_matrix, args.batch_size, device, logger
            )
        else:
            standard_metrics, predictions, labels = test_meta_model(
                model, test_dataset, distance_matrix, args, device, logger
            )
            probs = None

        # 噪声鲁棒性测试
        logger.info("\n" + "=" * 50)
        logger.info("2. 噪声鲁棒性测试")
        logger.info("=" * 50)
        args.noise_type = 'all'  # 测试所有噪声类型
        noise_results, clean_accuracy = test_noise_robustness(
            model, test_dataset, distance_matrix, args, device, logger
        )

        # 不同shot设置测试（仅对元学习模型）
        shot_results = None
        if args.model_type == 'meta':
            logger.info("\n" + "=" * 50)
            logger.info("3. 不同shot设置测试")
            logger.info("=" * 50)
            temp_k_shot = args.k_shot  # 保存原始设置
            shot_results, shot_values = test_different_shots(
                model, test_dataset, distance_matrix, args, device, logger
            )
            args.k_shot = temp_k_shot  # 恢复原始设置

        # 保存所有结果
        args.noise_results = noise_results
        args.shot_results = shot_results

        save_path = os.path.join(exp_dir, 'all_test_results.pth')
        torch.save({
            'standard_metrics': standard_metrics,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probs,
            'noise_results': noise_results,
            'clean_accuracy': clean_accuracy,
            'shot_results': shot_results,
            'shot_values': shot_values if args.model_type == 'meta' else None,
            'args': vars(args)
        }, save_path)
        logger.info(f"所有测试结果已保存至: {save_path}")

        # 可视化结果
        if args.visualize:
            logger.info("生成综合测试可视化...")
            visualize_results(model, test_dataset, standard_metrics, predictions, labels, args, exp_dir, logger)

    logger.info(f"Meta-HRRPNet测试完成: {args.exp_name}")


if __name__ == '__main__':
    main()