import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from dataset import prepare_datasets, TaskGenerator, HRRPTransform
from models import MDGN
from train import (
    MAMLPlusPlusTrainer as MAMLTrainer, test_model, shot_experiment,
    ablation_study_lambda, ablation_study_dynamic_graph,
    noise_robustness_experiment
)
from utils import (
    plot_learning_curve, plot_shot_curve, plot_confusion_matrix,
    visualize_features, visualize_dynamic_graph, visualize_attention,
    compute_metrics, log_metrics, create_experiment_log, find_latest_experiment
)


def parse_args():
    parser = argparse.ArgumentParser(description='Meta-Dynamic Graph Network for Few-Shot HRRP Recognition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'ablation'],
                        help='Run mode: train, test, or ablation study')
    parser.add_argument('--cv', type=int, default=0, choices=[0, 1, 2],
                        help='Cross-validation scheme index (0, 1, or 2)')
    parser.add_argument('--shot', type=int, default=5,
                        help='K-shot setting for few-shot learning')
    parser.add_argument('--query', type=int, default=None,
                        help='Query samples per class')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Task batch size')
    parser.add_argument('--vis', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (timestamp) for loading a previous model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Direct path to model file to load')
    return parser.parse_args()


def train(args, vis=False):
    """训练模型"""
    # 准备数据集
    train_dataset, test_dataset = prepare_datasets(scheme_idx=args.cv)

    # 检查是否有足够的样本
    if not train_dataset.samples or not test_dataset.samples:
        print("错误: 至少一个数据集没有找到有效样本。请检查数据路径和文件格式。")
        return None, 0

    # 设置K-shot和q_query
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.query is not None:
        Config.q_query = args.query

    # 计算每个类可用的最小样本数
    train_min_samples = float('inf')
    for class_idx in train_dataset.class_samples:
        samples_count = len(train_dataset.class_samples[class_idx])
        if samples_count < train_min_samples:
            train_min_samples = samples_count

    if train_min_samples == float('inf'):
        print("错误: 训练集没有可用样本。")
        return None, 0

    print(f"\n训练集中每个类的最小样本数: {train_min_samples}")

    # 自动调整k_shot和q_query
    total_needed = Config.k_shot + Config.q_query
    if total_needed > train_min_samples:
        old_k_shot = Config.k_shot
        old_q_query = Config.q_query

        # 重新分配样本
        if train_min_samples <= 1:
            Config.k_shot = 1
            Config.q_query = 0
        else:
            # 尝试保持至少一个查询样本
            Config.k_shot = min(old_k_shot, train_min_samples - 1)
            Config.q_query = train_min_samples - Config.k_shot

        print(
            f"警告: 样本不足，调整参数 k_shot: {old_k_shot} -> {Config.k_shot}, q_query: {old_q_query} -> {Config.q_query}")

    # 确保q_query至少为1
    if Config.q_query < 1:
        print("错误: 没有足够的样本用于查询集。每个类至少需要k_shot+1个样本。")
        if train_min_samples >= 2:
            Config.k_shot = train_min_samples - 1
            Config.q_query = 1
            print(f"自动调整为: k_shot={Config.k_shot}, q_query={Config.q_query}")
        else:
            return None, 0

    print(f"\n最终训练参数: {Config.k_shot}-shot, {Config.q_query}-query")

    # 创建任务生成器
    train_task_generator = TaskGenerator(train_dataset, n_way=Config.train_n_way, k_shot=Config.k_shot,
                                         q_query=Config.q_query)

    # 数据形状调试
    print("\n数据形状调试:")
    try:
        sample_task = train_task_generator.generate_task()
        support_x, support_y, query_x, query_y = sample_task
        print(f"支持集 X: 形状={support_x.shape}, 类型={support_x.dtype}")
        print(f"查询集 X: 形状={query_x.shape}, 类型={query_x.dtype}")

        # 检查是否为复数
        if torch.is_complex(support_x):
            print("警告: 支持集数据为复数类型，将转换为实数")
            support_x = torch.abs(support_x)

        # 检查值范围
        print(f"支持集数据范围: {support_x.min().item()} 到 {support_x.max().item()}")
    except Exception as e:
        print(f"调试时出错: {e}")

    val_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot,
                                       q_query=Config.q_query)

    # 创建模型和训练器
    model = MDGN(num_classes=Config.train_n_way)
    trainer = MAMLTrainer(model, Config.device)

    # 设置训练参数
    if args.epochs is not None:
        Config.max_epochs = args.epochs
    if args.batch is not None:
        Config.task_batch_size = args.batch

    # 创建实验日志
    create_experiment_log(Config)

    # 训练模型
    print(f"\n开始训练: {Config.k_shot}-shot, {Config.train_n_way}-way...")
    train_results = trainer.train(train_task_generator, val_task_generator, num_epochs=Config.max_epochs)

    # 绘制学习曲线
    if vis:
        plot_learning_curve(
            train_results['train_accs'],
            train_results['val_accs'],
            title=f"{Config.k_shot}-shot {Config.train_n_way}-way Learning Curve",
            save_path=os.path.join(Config.log_dir, 'learning_curve.png')
        )

    # 测试最佳模型
    print("\n测试最佳模型...")
    test_accuracy, test_ci, _ = test_model(model, val_task_generator, Config.device)
    print(f"最终测试准确率: {test_accuracy:.2f}% ± {test_ci:.2f}%")

    return model, test_accuracy


def test(args, model=None, vis=False):
    """测试模型"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return None
    # Try to find latest experiment if no model passed directly
    elif model is None:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    """测试模型"""
    # 准备数据集
    _, test_dataset = prepare_datasets(scheme_idx=args.cv)

    # 检查测试集样本
    if not test_dataset.samples:
        print("错误: 测试集没有找到有效样本。请检查数据路径和文件格式。")
        return None

    # 设置K-shot和q_query
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.query is not None:
        Config.q_query = args.query

    # 计算每个类可用的最小样本数
    test_min_samples = float('inf')
    for class_idx in test_dataset.class_samples:
        samples_count = len(test_dataset.class_samples[class_idx])
        if samples_count < test_min_samples:
            test_min_samples = samples_count

    if test_min_samples == float('inf'):
        print("错误: 测试集没有可用样本。")
        return None

    print(f"\n测试集中每个类的最小样本数: {test_min_samples}")

    # 自动调整k_shot和q_query
    total_needed = Config.k_shot + Config.q_query
    if total_needed > test_min_samples:
        old_k_shot = Config.k_shot
        old_q_query = Config.q_query

        # 重新分配样本
        if test_min_samples <= 1:
            Config.k_shot = 1
            Config.q_query = 0
        else:
            # 尝试保持至少一个查询样本
            Config.k_shot = min(old_k_shot, test_min_samples - 1)
            Config.q_query = test_min_samples - Config.k_shot

        print(
            f"警告: 样本不足，调整参数 k_shot: {old_k_shot} -> {Config.k_shot}, q_query: {old_q_query} -> {Config.q_query}")

    # 确保q_query至少为1
    if Config.q_query < 1:
        print("错误: 没有足够的样本用于查询集。每个类至少需要k_shot+1个样本。")
        if test_min_samples >= 2:
            Config.k_shot = test_min_samples - 1
            Config.q_query = 1
            print(f"自动调整为: k_shot={Config.k_shot}, q_query={Config.q_query}")
        else:
            return None

    print(f"\n最终测试参数: {Config.k_shot}-shot, {Config.q_query}-query")

    # 设置任务生成器
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot,
                                        q_query=Config.q_query)

    # Load model
    if model is None:
        model = MDGN(num_classes=Config.test_n_way)
        model = model.to(Config.device)

        # Prioritize directly specified model path
        if args.model_path and os.path.exists(args.model_path):
            model_path = args.model_path
        else:
            model_path = os.path.join(Config.save_dir, 'best_model.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model: {model_path}")
        else:
            print(f"Model not found: {model_path}. Please train first or provide correct path.")
            return None

    # 执行标准测试
    print(f"\n测试配置: {Config.k_shot}-shot, {Config.test_n_way}-way...")
    test_accuracy, test_ci, all_accuracies = test_model(model, test_task_generator, Config.device)
    print(f"测试准确率: {test_accuracy:.2f}% ± {test_ci:.2f}%")

    # 执行shot数量实验
    if vis:
        print("\n运行shot实验...")
        # 确定可行的shot大小
        max_shot = max(1, test_min_samples - 1)  # 至少保留1个样本用于查询

        # 调整shot范围不超过最大可用样本数
        feasible_shots = [s for s in [1, 2, 3, 5, 10, 20] if s < max_shot]
        if not feasible_shots:
            feasible_shots = [1]  # 至少使用1-shot

        print(f"可行的shot值: {feasible_shots}")

        shot_sizes, shot_results = shot_experiment(model, test_task_generator, Config.device, shot_sizes=feasible_shots)
        plot_shot_curve(
            shot_sizes, shot_results,
            title="Shot-Accuracy Curve",
            save_path=os.path.join(Config.log_dir, 'shot_curve.png')
        )

    return test_accuracy


def run_ablation_studies(args):
    """运行消融实验"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return
    # Try to find latest experiment
    else:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    """运行消融实验"""
    # 准备数据集
    _, test_dataset = prepare_datasets(scheme_idx=args.cv)

    # 检查测试集样本
    if not test_dataset.samples:
        print("错误: 测试集没有找到有效样本。请检查数据路径和文件格式。")
        return

    # 设置K-shot和q_query
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.query is not None:
        Config.q_query = args.query

    # 计算每个类可用的最小样本数
    test_min_samples = float('inf')
    for class_idx in test_dataset.class_samples:
        samples_count = len(test_dataset.class_samples[class_idx])
        if samples_count < test_min_samples:
            test_min_samples = samples_count

    if test_min_samples == float('inf'):
        print("错误: 测试集没有可用样本。")
        return

    print(f"\n测试集中每个类的最小样本数: {test_min_samples}")

    # 自动调整k_shot和q_query
    total_needed = Config.k_shot + Config.q_query
    if total_needed > test_min_samples:
        old_k_shot = Config.k_shot
        old_q_query = Config.q_query

        # 重新分配样本
        if test_min_samples <= 1:
            print("错误: 测试集样本不足，无法进行消融实验")
            return
        else:
            # 尝试保持至少一个查询样本
            Config.k_shot = min(old_k_shot, test_min_samples - 1)
            Config.q_query = test_min_samples - Config.k_shot

        print(
            f"警告: 样本不足，调整参数 k_shot: {old_k_shot} -> {Config.k_shot}, q_query: {old_q_query} -> {Config.q_query}")

    print(f"\n最终消融实验参数: {Config.k_shot}-shot, {Config.q_query}-query")

    # 设置任务生成器
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot,
                                        q_query=Config.q_query)

    # 加载模型
    model = MDGN(num_classes=Config.test_n_way)
    model = model.to(Config.device)
    model_path = os.path.join(Config.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"已加载模型: {model_path}")
    else:
        print(f"未找到模型: {model_path}。请先训练模型。")
        return

    # 运行lambda消融实验
    print("\n运行lambda消融实验...")
    lambda_values, lambda_results = ablation_study_lambda(model, test_task_generator, Config.device)
    plot_shot_curve(
        lambda_values, lambda_results,
        title="Lambda Mixing Coefficient Ablation Study",
        save_path=os.path.join(Config.log_dir, 'lambda_ablation.png')
    )

    # 运行动态图消融实验
    print("\n运行动态图消融实验...")
    graph_labels, graph_results = ablation_study_dynamic_graph(model, test_task_generator, Config.device)
    plt.figure(figsize=(10, 6))
    plt.bar(graph_labels, graph_results)
    plt.title("Dynamic vs Static Graph")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y')
    plt.savefig(os.path.join(Config.log_dir, 'dynamic_graph_ablation.png'))
    plt.close()

    # 运行噪声鲁棒性实验
    print("\n运行噪声鲁棒性实验...")
    noise_levels, noise_results = noise_robustness_experiment(model, test_task_generator, Config.device)
    plot_shot_curve(
        noise_levels, noise_results,
        title="Noise Robustness Experiment (SNR in dB)",
        save_path=os.path.join(Config.log_dir, 'noise_robustness.png')
    )


def main():
    args = parse_args()

    # Set random seed
    Config.set_seed()

    # Only create new directories in training mode
    if args.mode == 'train':
        Config.create_directories()

    try:
        # 根据模式执行
        if args.mode == 'train':
            model, _ = train(args, vis=args.vis)
            if model is not None:  # 只有当训练成功时才进行测试
                test(args, model=model, vis=args.vis)
        elif args.mode == 'test':
            test(args, vis=args.vis)
        elif args.mode == 'ablation':
            run_ablation_studies(args)
        else:
            print(f"未知模式: {args.mode}")
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查数据集路径和文件格式是否正确。")


if __name__ == "__main__":
    main()