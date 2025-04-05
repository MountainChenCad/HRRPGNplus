"""
消融实验 - 评估各个模块的贡献
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from scipy import stats

from data.dataset import HRRPDataset
from data.meta_dataset import MetaHRRPDataset
from models.baseline_gcn import HRRPGraphNet
from models.meta_graph_net import MetaHRRPNet
from trainers.standard_trainer import StandardTrainer
from trainers.maml_trainer import MAMLTrainer


class AblationExperiment:
    """
    消融实验 - 分析不同组件的贡献度

    参数:
    - config: 实验配置
    - data_root: 数据根目录
    - result_dir: 结果保存目录
    - seed: 随机种子，确保实验可重复
    """

    def __init__(self, config, data_root='data', result_dir='results', seed=3407):
        self.config = config
        self.data_root = data_root
        self.result_dir = os.path.join(result_dir, f'ablation_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
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

        # 定义消融组
        if hasattr(config, 'ABLATION_GROUPS'):
            self.ablation_groups = config.ABLATION_GROUPS
        else:
            # 默认消融组
            self.ablation_groups = {
                'base': {
                    'DYNAMIC_GRAPH': False,
                    'USE_META_CONV': False,
                    'USE_CURRICULUM': False,
                    'USE_META_ATTENTION': False
                },
                'dyn_graph': {
                    'DYNAMIC_GRAPH': True,
                    'USE_META_CONV': False,
                    'USE_CURRICULUM': False,
                    'USE_META_ATTENTION': False
                },
                'meta_conv': {
                    'DYNAMIC_GRAPH': True,
                    'USE_META_CONV': True,
                    'USE_CURRICULUM': False,
                    'USE_META_ATTENTION': False
                },
                'curriculum': {
                    'DYNAMIC_GRAPH': True,
                    'USE_META_CONV': True,
                    'USE_CURRICULUM': True,
                    'USE_META_ATTENTION': False
                },
                'full': {
                    'DYNAMIC_GRAPH': True,
                    'USE_META_CONV': True,
                    'USE_CURRICULUM': True,
                    'USE_META_ATTENTION': True
                }
            }

        self.logger.info(f"将进行以下消融组实验: {list(self.ablation_groups.keys())}")

    def _setup_logger(self):
        """设置日志记录器"""
        log_file = os.path.join(self.result_dir, 'ablation_exp.log')

        logger = logging.getLogger('ablation_experiment')
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

    def run_ablation(self, num_runs=5):
        """
        运行消融实验

        参数:
        - num_runs: 每个配置运行的次数（计算均值和标准差）
        """
        self.logger.info(f"开始消融实验，每个配置运行 {num_runs} 次...")

        # 存储所有结果
        all_results = {}

        # 对每个消融组进行实验
        for group_name, config_overrides in self.ablation_groups.items():
            self.logger.info(f"测试消融组: {group_name}")

            # 更新配置
            for key, value in config_overrides.items():
                setattr(self.config, key, value)

            group_results = []

            # 多次运行以获得统计显著性
            for run in range(num_runs):
                self.logger.info(f"运行 {run + 1}/{num_runs}")

                # 创建模型
                model = MetaHRRPNet(
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
                    k_shot=self.config.K_SHOT,
                    q_query=self.config.Q_QUERY,
                    num_tasks=self.config.TASKS_PER_EPOCH,
                    task_augment=True
                )

                meta_val_dataset = MetaHRRPDataset(
                    self.val_dataset,
                    n_way=self.config.N_WAY,
                    k_shot=self.config.K_SHOT,
                    q_query=self.config.Q_QUERY,
                    num_tasks=self.config.EVAL_TASKS
                )

                meta_test_dataset = MetaHRRPDataset(
                    self.test_dataset,
                    n_way=self.config.N_WAY,
                    k_shot=self.config.K_SHOT,
                    q_query=self.config.Q_QUERY,
                    num_tasks=self.config.EVAL_TASKS
                )

                # 创建训练器
                trainer = MAMLTrainer(
                    model,
                    meta_train_dataset,
                    meta_val_dataset,
                    meta_test_dataset,
                    self.config,
                    self.logger
                )

                # 训练模型
                history, metrics = trainer.train()

                # 记录结果
                result = {
                    'history': history,
                    'metrics': metrics,
                    'config': {k: getattr(self.config, k) for k in config_overrides.keys()}
                }

                group_results.append(result)

                # 输出当前运行的结果
                self.logger.info(f"运行 {run + 1} 结果: 准确率 = {metrics['accuracy']:.2f}%, "
                                 f"F1 = {metrics['f1']:.2f}%")

            # 保存该组的所有运行结果
            all_results[group_name] = group_results

            # 计算平均性能和标准差
            accuracies = [r['metrics']['accuracy'] for r in group_results]
            f1_scores = [r['metrics']['f1'] for r in group_results]

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)

            self.logger.info(f"{group_name} 平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
            self.logger.info(f"{group_name} 平均F1分数: {mean_f1:.2f}% ± {std_f1:.2f}%")

        # 保存所有结果
        save_path = os.path.join(self.result_dir, 'ablation_results.pth')
        torch.save(all_results, save_path)

        # 可视化结果
        self._visualize_ablation_results(all_results)

        # 进行统计显著性检验
        self._statistical_significance_test(all_results)

        self.logger.info(f"消融实验完成，结果已保存至 {save_path}")

        return all_results

    def _visualize_ablation_results(self, all_results):
        """可视化消融实验结果"""
        # 准备数据
        groups = list(all_results.keys())
        mean_accuracies = []
        std_accuracies = []

        for group in groups:
            accuracies = [r['metrics']['accuracy'] for r in all_results[group]]
            mean_accuracies.append(np.mean(accuracies))
            std_accuracies.append(np.std(accuracies))

        # 绘制柱状图
        plt.figure(figsize=(12, 6))

        x = np.arange(len(groups))
        plt.bar(x, mean_accuracies, yerr=std_accuracies, align='center', alpha=0.7,
                capsize=10, error_kw=dict(capthick=2))

        plt.axhline(y=mean_accuracies[0], color='r', linestyle='--', alpha=0.5,
                    label=f'Baseline ({mean_accuracies[0]:.1f}%)')

        plt.xlabel('Model Variant')
        plt.ylabel('Accuracy (%)')
        plt.title('Ablation Study Results')
        plt.xticks(x, groups)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend()

        # 添加数值标签
        for i, v in enumerate(mean_accuracies):
            plt.text(i, v + std_accuracies[i] + 0.5, f"{v:.1f}%",
                     ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.result_dir, 'ablation_results.png')
        plt.savefig(save_path, dpi=300)

        # 计算每个组件的贡献度
        if len(groups) >= 2:  # 至少需要基线和一个变种
            plt.figure(figsize=(10, 6))

            contributions = []
            labels = []

            baseline_acc = mean_accuracies[0]
            full_acc = mean_accuracies[-1]
            total_improvement = full_acc - baseline_acc

            if total_improvement > 0:
                # 计算每个组件的增益
                for i in range(1, len(groups)):
                    prev_acc = mean_accuracies[i - 1]
                    curr_acc = mean_accuracies[i]
                    component_gain = curr_acc - prev_acc
                    component_contribution = component_gain / total_improvement * 100

                    contributions.append(component_contribution)
                    labels.append(f"{groups[i]} (+{component_contribution:.1f}%)")

                # 绘制饼图
                plt.pie(contributions, labels=labels, autopct='%1.1f%%',
                        startangle=90, shadow=True)
                plt.axis('equal')
                plt.title('Component Contribution to Overall Improvement')

                # 保存图像
                save_path = os.path.join(self.result_dir, 'component_contributions.png')
                plt.savefig(save_path, dpi=300)

        self.logger.info(f"可视化结果已保存至 {self.result_dir}")

    def _statistical_significance_test(self, all_results):
        """进行统计显著性检验"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("统计显著性检验 (t-test)")
        self.logger.info("=" * 50)

        groups = list(all_results.keys())

        # 创建结果表格
        p_values = np.zeros((len(groups), len(groups)))

        for i, group1 in enumerate(groups):
            accs1 = [r['metrics']['accuracy'] for r in all_results[group1]]
            for j, group2 in enumerate(groups):
                if i == j:
                    p_values[i, j] = 1.0  # 自身比较
                else:
                    accs2 = [r['metrics']['accuracy'] for r in all_results[group2]]
                    _, p_value = stats.ttest_ind(accs1, accs2, equal_var=False)
                    p_values[i, j] = p_value

        # 打印p值表格
        self.logger.info("P值表格 (行 vs 列):")
        header = "\t" + "\t".join(groups)
        self.logger.info(header)

        for i, group in enumerate(groups):
            row = group + "\t" + "\t".join([f"{p:.4f}" for p in p_values[i]])
            self.logger.info(row)

        # 分析显著性
        alpha = self.config.ALPHA_VALUE if hasattr(self.config, 'ALPHA_VALUE') else 0.05
        self.logger.info(f"\n显著性级别: α = {alpha}")

        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i != j:
                    p = p_values[i, j]
                    significant = p < alpha
                    if significant:
                        self.logger.info(f"{group1} vs {group2}: p={p:.4f} < {alpha} (显著差异)")
                    else:
                        self.logger.info(f"{group1} vs {group2}: p={p:.4f} >= {alpha} (无显著差异)")

        self.logger.info("=" * 50)

        # 绘制热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_values, annot=True, fmt=".4f", cmap="YlGnBu",
                    xticklabels=groups, yticklabels=groups)
        plt.title('P-values for Pairwise Comparisons')
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.result_dir, 'pvalue_heatmap.png')
        plt.savefig(save_path, dpi=300)