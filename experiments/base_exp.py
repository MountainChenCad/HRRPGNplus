"""
基础比较实验 - 对比标准模型与元学习模型
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import logging
from datetime import datetime

from data.dataset import HRRPDataset
from data.meta_dataset import MetaHRRPDataset, TaskSampler
from models.baseline_gcn import HRRPGraphNet
from models.meta_graph_net import MetaHRRPNet
from trainers.standard_trainer import StandardTrainer
from trainers.maml_trainer import MAMLTrainer
from utils.visualization import plot_confusion_matrix, plot_learning_curves, plot_tsne


class BaseExperiment:
    """
    基础实验 - 对比标准方法与元学习方法

    参数:
    - config: 实验配置
    - data_root: 数据根目录
    - result_dir: 结果保存目录
    - seed: 随机种子，确保实验可重复
    """

    def __init__(self, config, data_root='data', result_dir='results', seed=3407):
        self.config = config
        self.data_root = data_root
        self.result_dir = os.path.join(result_dir, f'base_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.result_dir, exist_ok=True)
        self.seed = seed

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 设置日志
        self.logger = self._setup_logger()

        # 加载数据集
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_datasets()

        # 记录配置
        self.logger.info(f"实验配置: {config.get_config_dict()}")
        self.logger.info(f"数据集统计: 训练集 {len(self.train_dataset)} 样本, "
                         f"验证集 {len(self.val_dataset)} 样本, "
                         f"测试集 {len(self.test_dataset)} 样本")

    def _setup_logger(self):
        """设置日志记录器"""
        log_file = os.path.join(self.result_dir, 'experiment.log')

        logger = logging.getLogger('base_experiment')
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

    def _load_datasets(self):
        """加载数据集"""
        # 加载训练、验证和测试数据集
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val')
        test_dir = os.path.join(self.data_root, 'test')

        train_dataset = HRRPDataset(train_dir)
        val_dataset = HRRPDataset(val_dir)
        test_dataset = HRRPDataset(test_dir)

        return train_dataset, val_dataset, test_dataset

    def run_baseline(self):
        """运行基线模型（原始HRRPGraphNet）"""
        self.logger.info("开始运行基线模型...")

        # 创建模型
        num_classes = len(self.train_dataset.classes)
        baseline_model = HRRPGraphNet(num_classes=num_classes)

        # 创建训练器
        trainer = StandardTrainer(
            baseline_model,
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.config,
            self.logger
        )

        # 训练模型
        history, metrics = trainer.train()

        # 保存结果
        results = {
            'model': 'HRRPGraphNet',
            'history': history,
            'metrics': metrics
        }

        # 保存模型和结果
        save_path = os.path.join(self.result_dir, 'baseline_results.pth')
        torch.save(results, save_path)

        self.logger.info(f"基线模型结果已保存至 {save_path}")

        return results

    def run_meta_model(self):
        """运行元学习模型（Meta-HRRPNet）"""
        self.logger.info("开始运行元学习模型...")

        # 创建元学习数据集
        self.meta_train_dataset = MetaHRRPDataset(
            self.train_dataset,
            n_way=self.config.N_WAY,
            k_shot=self.config.K_SHOT,
            q_query=self.config.Q_QUERY,
            num_tasks=self.config.TASKS_PER_EPOCH,
            task_augment=True
        )

        self.meta_val_dataset = MetaHRRPDataset(
            self.val_dataset,
            n_way=self.config.N_WAY,
            k_shot=self.config.K_SHOT,
            q_query=self.config.Q_QUERY,
            num_tasks=self.config.EVAL_TASKS
        )

        self.meta_test_dataset = MetaHRRPDataset(
            self.test_dataset,
            n_way=self.config.N_WAY,
            k_shot=self.config.K_SHOT,
            q_query=self.config.Q_QUERY,
            num_tasks=self.config.EVAL_TASKS
        )

        # 创建模型
        meta_model = MetaHRRPNet(
            num_classes=self.config.N_WAY,
            feature_dim=self.config.FEATURE_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            use_dynamic_graph=self.config.DYNAMIC_GRAPH,
            use_meta_attention=self.config.USE_META_ATTENTION,
            alpha=self.config.ALPHA,
            num_heads=self.config.NUM_HEADS
        )

        # 创建训练器
        trainer = MAMLTrainer(
            meta_model,
            self.meta_train_dataset,
            self.meta_val_dataset,
            self.meta_test_dataset,
            self.config,
            self.logger
        )

        # 训练模型
        history, metrics = trainer.train()

        # 保存结果
        results = {
            'model': 'Meta-HRRPNet',
            'history': history,
            'metrics': metrics
        }

        # 保存模型和结果
        save_path = os.path.join(self.result_dir, 'meta_model_results.pth')
        torch.save(results, save_path)

        self.logger.info(f"元学习模型结果已保存至 {save_path}")

        return results

    def run_shot_comparison(self):
        """在不同shot设置下比较性能"""
        self.logger.info("开始比较不同shot设置下的性能...")

        shot_values = self.config.get_shot_values()
        baseline_results = {}
        meta_results = {}

        for k_shot in shot_values:
            self.logger.info(f"评估 {k_shot}-shot 设置")

            # 更新配置
            self.config.K_SHOT = k_shot

            # 运行基线
            self.logger.info(f"运行基线模型 ({k_shot}-shot)...")
            # 为基线模型限制训练样本
            limited_train_dataset = self._create_limited_dataset(k_shot)

            # 创建模型
            num_classes = len(self.train_dataset.classes)
            baseline_model = HRRPGraphNet(num_classes=num_classes)

            # 创建训练器
            trainer = StandardTrainer(
                baseline_model,
                limited_train_dataset,
                self.val_dataset,
                self.test_dataset,
                self.config,
                self.logger
            )

            # 训练并测试
            _, baseline_metrics = trainer.train()
            baseline_results[k_shot] = baseline_metrics

            # 运行元学习模型
            self.logger.info(f"运行元学习模型 ({k_shot}-shot)...")
            meta_model = MetaHRRPNet(
                num_classes=self.config.N_WAY,
                feature_dim=self.config.FEATURE_DIM,
                hidden_dim=self.config.HIDDEN_DIM,
                use_dynamic_graph=self.config.DYNAMIC_GRAPH,
                use_meta_attention=self.config.USE_META_ATTENTION,
                alpha=self.config.ALPHA,
                num_heads=self.config.NUM_HEADS
            )

            # 创建元学习数据集
            meta_train_dataset = MetaHRRPDataset(
                self.train_dataset,
                n_way=self.config.N_WAY,
                k_shot=k_shot,
                q_query=self.config.Q_QUERY,
                num_tasks=self.config.TASKS_PER_EPOCH
            )

            meta_val_dataset = MetaHRRPDataset(
                self.val_dataset,
                n_way=self.config.N_WAY,
                k_shot=k_shot,
                q_query=self.config.Q_QUERY,
                num_tasks=self.config.EVAL_TASKS
            )

            meta_test_dataset = MetaHRRPDataset(
                self.test_dataset,
                n_way=self.config.N_WAY,
                k_shot=k_shot,
                q_query=self.config.Q_QUERY,
                num_tasks=self.config.EVAL_TASKS
            )

            # 创建训练器
            trainer = MAMLTrainer(
                meta_model,
                meta_train_dataset,
                meta_val_dataset,
                meta_test_dataset,
                self.config,
                self.logger
            )

            # 训练并测试
            _, meta_metrics = trainer.train()
            meta_results[k_shot] = meta_metrics

        # 保存结果
        shot_comparison = {
            'shot_values': shot_values,
            'baseline_results': baseline_results,
            'meta_results': meta_results
        }

        save_path = os.path.join(self.result_dir, 'shot_comparison.pth')
        torch.save(shot_comparison, save_path)

        # 绘制比较图
        self._plot_shot_comparison(shot_values, baseline_results, meta_results)

        return shot_comparison

    def _create_limited_dataset(self, k_shot):
        """创建限制每类样本数的数据集"""
        # 获取每个类别的样本计数
        class_counts = self.train_dataset.get_class_count()

        # 创建新的文件列表，每类限制k_shot个样本
        limited_files = []
        for cls in self.train_dataset.classes:
            # 获取该类的文件路径
            class_files = self.train_dataset.samples_by_class[cls]

            # 随机选择k_shot个样本
            if len(class_files) > k_shot:
                selected_files = np.random.choice(class_files, k_shot, replace=False)
            else:
                selected_files = class_files

            # 提取文件名
            selected_filenames = [os.path.basename(f) for f in selected_files]
            limited_files.extend(selected_filenames)

        # 创建新的数据集实例
        limited_dataset = HRRPDataset(self.train_dataset.root_dir)

        # 覆盖文件列表
        limited_dataset.file_list = limited_files

        self.logger.info(f"创建了限制每类 {k_shot} 样本的数据集，共 {len(limited_files)} 个样本")

        return limited_dataset

    def _plot_shot_comparison(self, shot_values, baseline_results, meta_results):
        """绘制不同shot设置下的性能比较图"""
        plt.figure(figsize=(10, 6))

        # 提取准确率
        baseline_acc = [baseline_results[k]['accuracy'] for k in shot_values]
        meta_acc = [meta_results[k]['accuracy'] for k in shot_values]

        plt.plot(shot_values, baseline_acc, 'o-', label='HRRPGraphNet')
        plt.plot(shot_values, meta_acc, 's-', label='Meta-HRRPNet')

        plt.xlabel('每类样本数 (K-shot)')
        plt.ylabel('准确率 (%)')
        plt.title('不同样本数下的模型性能比较')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.result_dir, 'shot_comparison.png')
        plt.savefig(save_path, dpi=300)
        self.logger.info(f"比较图已保存至 {save_path}")

    def run_comparison(self):
        """运行基线和元学习模型的完整比较"""
        self.logger.info("开始运行模型比较实验...")

        # 运行基线模型
        baseline_results = self.run_baseline()

        # 运行元学习模型
        meta_results = self.run_meta_model()

        # 可视化比较
        self._visualize_comparison(baseline_results, meta_results)

        # 不同shot设置的比较
        shot_comparison = self.run_shot_comparison()

        # 汇总所有结果
        all_results = {
            'baseline': baseline_results,
            'meta': meta_results,
            'shot_comparison': shot_comparison
        }

        # 保存所有结果
        save_path = os.path.join(self.result_dir, 'all_results.pth')
        torch.save(all_results, save_path)

        self.logger.info(f"所有实验结果已保存至 {save_path}")

        # 打印比较总结
        self._print_comparison_summary(baseline_results, meta_results)

        return all_results

    def _visualize_comparison(self, baseline_results, meta_results):
        """可视化两个模型的比较结果"""
        # 创建可视化目录
        vis_dir = os.path.join(self.result_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # 学习曲线比较
        baseline_history = baseline_results['history']
        meta_history = meta_results['history']

        # 训练损失比较
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(baseline_history['train_loss'], label='HRRPGraphNet')
        plt.plot(meta_history['train_loss'], label='Meta-HRRPNet')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 验证准确率比较
        plt.subplot(1, 2, 2)
        plt.plot(baseline_history['val_acc'], label='HRRPGraphNet')
        plt.plot(meta_history['val_acc'], label='Meta-HRRPNet')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Validation Accuracy Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'learning_curves_comparison.png'), dpi=300)

        # 类别准确率比较
        baseline_class_acc = baseline_results['metrics']['class_accuracies']
        meta_class_acc = meta_results['metrics']['class_accuracies']

        # 确保两个字典有相同的键
        all_classes = sorted(set(baseline_class_acc.keys()) | set(meta_class_acc.keys()))

        baseline_acc_list = [baseline_class_acc.get(cls, 0) for cls in all_classes]
        meta_acc_list = [meta_class_acc.get(cls, 0) for cls in all_classes]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(all_classes))
        width = 0.35

        plt.bar(x - width / 2, baseline_acc_list, width, label='HRRPGraphNet')
        plt.bar(x + width / 2, meta_acc_list, width, label='Meta-HRRPNet')

        plt.xlabel('类别')
        plt.ylabel('准确率 (%)')
        plt.title('各类别准确率比较')
        plt.xticks(x, [str(cls) for cls in all_classes])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'class_accuracy_comparison.png'), dpi=300)

        self.logger.info(f"可视化结果已保存至 {vis_dir}")

    def _print_comparison_summary(self, baseline_results, meta_results):
        """打印比较结果总结"""
        baseline_metrics = baseline_results['metrics']
        meta_metrics = meta_results['metrics']

        self.logger.info("\n" + "=" * 50)
        self.logger.info("模型性能比较总结")
        self.logger.info("=" * 50)

        self.logger.info(f"HRRPGraphNet (基线):")
        self.logger.info(f"  - 准确率: {baseline_metrics['accuracy']:.2f}%")
        self.logger.info(f"  - 精确率: {baseline_metrics['precision']:.2f}%")
        self.logger.info(f"  - 召回率: {baseline_metrics['recall']:.2f}%")
        self.logger.info(f"  - F1 分数: {baseline_metrics['f1']:.2f}%")

        self.logger.info(f"\nMeta-HRRPNet (提出的方法):")
        self.logger.info(f"  - 准确率: {meta_metrics['accuracy']:.2f}%")
        self.logger.info(f"  - 精确率: {meta_metrics['precision']:.2f}%")
        self.logger.info(f"  - 召回率: {meta_metrics['recall']:.2f}%")
        self.logger.info(f"  - F1 分数: {meta_metrics['f1']:.2f}%")

        # 计算改进百分比
        acc_improvement = (meta_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100
        f1_improvement = (meta_metrics['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100

        self.logger.info(f"\n改进:")
        self.logger.info(f"  - 准确率提升: {acc_improvement:.2f}%")
        self.logger.info(f"  - F1 分数提升: {f1_improvement:.2f}%")
        self.logger.info("=" * 50)