import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from dataset import prepare_datasets, TaskGenerator, HRRPTransform
from models import MDGN
from train import (
    MAMLTrainer, test_model, shot_experiment,
    ablation_study_lambda, ablation_study_dynamic_graph,
    noise_robustness_experiment
)
from utils import (
    plot_learning_curve, plot_shot_curve, plot_confusion_matrix,
    visualize_features, visualize_dynamic_graph, visualize_attention,
    compute_metrics, log_metrics, create_experiment_log
)


def parse_args():
    parser = argparse.ArgumentParser(description='Meta-Dynamic Graph Network for Few-Shot HRRP Recognition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'ablation'],
                        help='Run mode: train, test, or ablation study')
    parser.add_argument('--cv', type=int, default=0, choices=[0, 1, 2],
                        help='Cross-validation scheme index (0, 1, or 2)')
    parser.add_argument('--shot', type=int, default=5,
                        help='K-shot setting for few-shot learning')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Task batch size')
    parser.add_argument('--vis', action='store_true',
                        help='Enable visualization')
    return parser.parse_args()


def train(args, vis=False):
    """训练模型"""
    # 准备数据集
    train_dataset, test_dataset = prepare_datasets(scheme_idx=args.cv)

    # 设置K-shot
    if args.shot is not None:
        Config.k_shot = args.shot

    # 设置任务生成器
    train_task_generator = TaskGenerator(train_dataset, k_shot=Config.k_shot)
    val_task_generator = TaskGenerator(test_dataset, k_shot=Config.k_shot)

    # 创建模型和训练器
    num_classes = Config.n_way
    model = MDGN(num_classes=num_classes)
    trainer = MAMLTrainer(model, Config.device)

    # 设置训练参数
    if args.epochs is not None:
        Config.max_epochs = args.epochs
    if args.batch is not None:
        Config.task_batch_size = args.batch

    # 创建实验日志
    create_experiment_log(Config)

    # 训练模型
    print(f"Training with {Config.k_shot}-shot, {Config.n_way}-way configuration...")
    train_results = trainer.train(train_task_generator, val_task_generator, num_epochs=Config.max_epochs)

    # 绘制学习曲线
    if vis:
        plot_learning_curve(
            train_results['train_accs'],
            train_results['val_accs'],
            title=f"{Config.k_shot}-shot {Config.n_way}-way Learning Curve",
            save_path=os.path.join(Config.log_dir, 'learning_curve.png')
        )

    # 测试最佳模型
    print("\nTesting best model...")
    test_accuracy, test_ci, _ = test_model(model, val_task_generator, Config.device)
    print(f"Final Test Accuracy: {test_accuracy:.2f}% ± {test_ci:.2f}%")

    return model, test_accuracy


def test(args, model=None, vis=False):
    """测试模型"""
    # 准备数据集
    _, test_dataset = prepare_datasets(scheme_idx=args.cv)

    # 设置K-shot
    if args.shot is not None:
        Config.k_shot = args.shot

    # 设置任务生成器
    test_task_generator = TaskGenerator(test_dataset, k_shot=Config.k_shot)

    # 加载模型
    if model is None:
        num_classes = Config.n_way
        model = MDGN(num_classes=num_classes)
        model = model.to(Config.device)
        model_path = os.path.join(Config.save_dir, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}. Please train first.")
            return None

    # 执行标准测试
    print(f"\nTesting with {Config.k_shot}-shot, {Config.n_way}-way configuration...")
    test_accuracy, test_ci, all_accuracies = test_model(model, test_task_generator, Config.device)
    print(f"Test Accuracy: {test_accuracy:.2f}% ± {test_ci:.2f}%")

    # 执行shot数量实验
    if vis:
        print("\nRunning shot experiment...")
        shot_sizes, shot_results = shot_experiment(model, test_task_generator, Config.device)
        plot_shot_curve(
            shot_sizes, shot_results,
            title="Shot-Accuracy Curve",
            save_path=os.path.join(Config.log_dir, 'shot_curve.png')
        )

    return test_accuracy


def run_ablation_studies(args):
    """运行消融实验"""
    # 准备数据集
    _, test_dataset = prepare_datasets(scheme_idx=args.cv)

    # 设置K-shot
    if args.shot is not None:
        Config.k_shot = args.shot

    # 设置任务生成器
    test_task_generator = TaskGenerator(test_dataset, k_shot=Config.k_shot)

    # 加载模型
    num_classes = Config.n_way
    model = MDGN(num_classes=num_classes)
    model = model.to(Config.device)
    model_path = os.path.join(Config.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Please train first.")
        return

    # 运行lambda消融实验
    print("\nRunning lambda ablation study...")
    lambda_values, lambda_results = ablation_study_lambda(model, test_task_generator, Config.device)
    plot_shot_curve(
        lambda_values, lambda_results,
        title="Lambda Mixing Coefficient Ablation Study",
        save_path=os.path.join(Config.log_dir, 'lambda_ablation.png')
    )

    # 运行动态图消融实验
    print("\nRunning dynamic graph ablation study...")
    graph_labels, graph_results = ablation_study_dynamic_graph(model, test_task_generator, Config.device)
    plt.figure(figsize=(10, 6))
    plt.bar(graph_labels, graph_results)
    plt.title("Dynamic vs Static Graph")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y')
    plt.savefig(os.path.join(Config.log_dir, 'dynamic_graph_ablation.png'))
    plt.close()

    # 运行噪声鲁棒性实验
    print("\nRunning noise robustness experiment...")
    noise_levels, noise_results = noise_robustness_experiment(model, test_task_generator, Config.device)
    plot_shot_curve(
        noise_levels, noise_results,
        title="Noise Robustness Experiment (SNR in dB)",
        save_path=os.path.join(Config.log_dir, 'noise_robustness.png')
    )


def main():
    args = parse_args()

    # 设置随机种子
    Config.set_seed()

    # 创建必要的目录
    Config.create_directories()

    # 根据模式执行
    if args.mode == 'train':
        model, _ = train(args, vis=args.vis)
        test(args, model=model, vis=args.vis)
    elif args.mode == 'test':
        test(args, vis=args.vis)
    elif args.mode == 'ablation':
        run_ablation_studies(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()