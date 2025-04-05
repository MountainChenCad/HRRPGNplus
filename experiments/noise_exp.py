"""
噪声鲁棒性实验 - 测试模型在不同噪声条件下的性能
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

from data.dataset import HRRPDataset
from data.meta_dataset import MetaHRRPDataset
from models.baseline_gcn import HRRPGraphNet
from models.meta_graph_net import MetaHRRPNet
from trainers.standard_trainer import StandardTrainer
from trainers.maml_trainer import MAMLTrainer


class NoiseExperiment:
    """
    噪声鲁棒性实验 - 评估不同噪声条件下的模型性能

    参数:
    - config: 实验配置
    - data_root: 数据根目录
    - result_dir: 结果保存目录
    - seed: 随机种子，确保实验可重复
    """

    def __init__(self, config, data_root='data', result_dir='results', seed=3407):
        self.config = config
        self.data_root = data_root
        self.result_dir = os.path.join(result_dir, f'noise_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.result_dir, exist_ok=True)
        self.seed = seed

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 设置日志
        self.logger = self._setup_logger()

        # 加载数据集
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_datasets()

        # 定义噪声类型和信噪比
        self.noise_types = getattr(config, 'NOISE_TYPES', ['gaussian', 'impulse', 'speckle'])
        self.snr_levels = getattr(config, 'SNR_LEVELS', [-5, 0, 5, 10, 15, 20])

        self.logger.info(f"实验配置: {config.get_config_dict()}")
        self.logger.info(f"噪声类型: {self.noise_types}")
        self.logger.info(f"信噪比(dB): {self.snr_levels}")

    def _setup_logger(self):
        """设置日志记录器"""
        log_file = os.path.join(self.result_dir, 'noise_exp.log')

        logger = logging.getLogger('noise_experiment')
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

    def _create_noisy_dataset(self, dataset, noise_type, snr):
        """创建带噪声的数据集"""
        self.logger.info(f"创建 {noise_type} 噪声数据集，SNR={snr}dB")

        # 获取噪声参数
        if hasattr(self.config, 'get_noise_params'):
            noise_params = self.config.get_noise_params(noise_type, snr)
        else:
            # 默认噪声参数
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

    def run_noise_experiment(self):
        """运行噪声鲁棒性实验"""
        self.logger.info("开始噪声鲁棒性实验...")

        # 加载训练好的模型
        standard_model_path = os.path.join(self.config.SAVE_DIR, "best_model_standard.pth")
        meta_model_path = os.path.join(self.config.SAVE_DIR, "best_meta_model.pth")

        # 检查模型文件是否存在
        if not os.path.exists(standard_model_path) or not os.path.exists(meta_model_path):
            self.logger.warning("模型文件不存在，将训练新模型")
            self._train_models()

        # 创建结果字典
        results = {
            'standard': {noise_type: {} for noise_type in self.noise_types},
            'meta': {noise_type: {} for noise_type in self.noise_types}
        }

        # 加载标准模型
        self.logger.info("加载标准模型...")
        standard_model = HRRPGraphNet(num_classes=len(self.train_dataset.classes))
        standard_model.to(self.config.DEVICE)
        standard_checkpoint = torch.load(standard_model_path, map_location=self.config.DEVICE)
        standard_model.load_state_dict(standard_checkpoint['model_state_dict'])
        standard_model.eval()

        # 加载元学习模型
        self.logger.info("加载元学习模型...")
        meta_model = MetaHRRPNet(num_classes=self.config.N_WAY,
                                 feature_dim=self.config.FEATURE_DIM,
                                 use_dynamic_graph=self.config.DYNAMIC_GRAPH,
                                 use_meta_attention=self.config.USE_META_ATTENTION)
        meta_model.to(self.config.DEVICE)
        meta_checkpoint = torch.load(meta_model_path, map_location=self.config.DEVICE)
        meta_model.load_state_dict(meta_checkpoint['model_state_dict'])
        meta_model.eval()

        # 生成距离矩阵
        distance_matrix = self._generate_distance_matrix(self.config.FEATURE_DIM).to(self.config.DEVICE)

        # 在干净数据上测试基准性能
        self.logger.info("测试干净数据上的性能...")
        clean_standard_acc = self._evaluate_standard_model(standard_model, self.test_dataset, distance_matrix)
        clean_meta_acc = self._evaluate_meta_model(meta_model, self.test_dataset, distance_matrix)

        self.logger.info(f"干净数据上的性能 - 标准模型: {clean_standard_acc:.2f}%, 元学习模型: {clean_meta_acc:.2f}%")

        # 对每种噪声类型和信噪比进行测试
        for noise_type in self.noise_types:
            for snr in self.snr_levels:
                self.logger.info(f"测试 {noise_type} 噪声, SNR={snr}dB")

                # 创建带噪声的测试数据集
                noisy_test_dataset = self._create_noisy_dataset(self.test_dataset, noise_type, snr)

                # 测试标准模型
                standard_acc = self._evaluate_standard_model(standard_model, noisy_test_dataset, distance_matrix)
                # 计算鲁棒性比例
                standard_robustness = (standard_acc / clean_standard_acc) * 100

                # 测试元学习模型
                meta_acc = self._evaluate_meta_model(meta_model, noisy_test_dataset, distance_matrix)
                # 计算鲁棒性比例
                meta_robustness = (meta_acc / clean_meta_acc) * 100

                # 记录结果
                results['standard'][noise_type][snr] = {
                    'accuracy': standard_acc,
                    'robustness': standard_robustness
                }

                results['meta'][noise_type][snr] = {
                    'accuracy': meta_acc,
                    'robustness': meta_robustness
                }

                self.logger.info(f"结果 - 标准模型: {standard_acc:.2f}% ({standard_robustness:.2f}%), "
                                 f"元学习模型: {meta_acc:.2f}% ({meta_robustness:.2f}%)")

        # 保存结果
        results['clean'] = {
            'standard': clean_standard_acc,
            'meta': clean_meta_acc
        }

        save_path = os.path.join(self.result_dir, 'noise_results.pth')
        torch.save(results, save_path)

        # 可视化结果
        self._visualize_noise_results(results)

        self.logger.info(f"噪声实验完成，结果已保存至 {save_path}")

        return results

    def _generate_distance_matrix(self, N):
        """生成距离矩阵"""
        distance_matrix = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = 1 / (abs(i - j) + 1)
        return distance_matrix

    def _train_models(self):
        """训练标准模型和元学习模型"""
        # 训练标准模型
        self.logger.info("训练标准模型...")
        standard_model = HRRPGraphNet(num_classes=len(self.train_dataset.classes))

        standard_trainer = StandardTrainer(
            standard_model,
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.config,
            self.logger
        )

        standard_trainer.train()

        # 训练元学习模型
        self.logger.info("训练元学习模型...")
        meta_model = MetaHRRPNet(
            num_classes=self.config.N_WAY,
            feature_dim=self.config.FEATURE_DIM,
            use_dynamic_graph=self.config.DYNAMIC_GRAPH,
            use_meta_attention=self.config.USE_META_ATTENTION
        )

        meta_train_dataset = MetaHRRPDataset(
            self.train_dataset,
            n_way=self.config.N_WAY,
            k_shot=self.config.K_SHOT,
            q_query=self.config.Q_QUERY,
            num_tasks=self.config.TASKS_PER_EPOCH
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

        meta_trainer = MAMLTrainer(
            meta_model,
            meta_train_dataset,
            meta_val_dataset,
            meta_test_dataset,
            self.config,
            self.logger
        )

        meta_trainer.train()

    def _evaluate_standard_model(self, model, dataset, distance_matrix):
        """评估标准模型性能"""
        model.eval()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.config.DEVICE), targets.to(self.config.DEVICE)

                # 前向传播
                outputs = model(data, distance_matrix)

                # 统计准确率
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def _evaluate_meta_model(self, model, dataset, distance_matrix):
        """评估元学习模型性能"""
        model.eval()

        # 创建元数据集
        meta_dataset = MetaHRRPDataset(
            dataset,
            n_way=self.config.N_WAY,
            k_shot=self.config.K_SHOT,
            q_query=self.config.Q_QUERY,
            num_tasks=self.config.EVAL_TASKS
        )

        # 采样任务进行评估
        sampler = torch.utils.data.DataLoader(
            meta_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        all_accuracies = []

        with torch.no_grad():
            for task_batch in tqdm(sampler, desc="Evaluating meta-model"):
                # 提取支持集和查询集
                support_x = task_batch['support_x'][0].to(self.config.DEVICE)
                support_y = task_batch['support_y'][0].to(self.config.DEVICE)
                query_x = task_batch['query_x'][0].to(self.config.DEVICE)
                query_y = task_batch['query_y'][0].to(self.config.DEVICE)

                # 快速适应
                for step in range(self.config.INNER_STEPS):
                    if hasattr(model, 'use_dynamic_graph') and model.use_dynamic_graph:
                        logits, _ = model(support_x, distance_matrix)
                    else:
                        logits = model(support_x, distance_matrix)

                    loss = torch.nn.functional.cross_entropy(logits, support_y)

                    # 计算梯度
                    grads = torch.autograd.grad(
                        loss,
                        model.parameters(),
                        create_graph=False
                    )

                    # 更新权重
                    for param, grad in zip(model.parameters(), grads):
                        param.data = param.data - self.config.INNER_LR * grad

                # 在查询集上评估
                if hasattr(model, 'use_dynamic_graph') and model.use_dynamic_graph:
                    query_logits, _ = model(query_x, distance_matrix)
                else:
                    query_logits = model(query_x, distance_matrix)

                # 计算准确率
                _, predicted = query_logits.max(1)
                accuracy = 100.0 * predicted.eq(query_y).float().mean().item()
                all_accuracies.append(accuracy)

                # 恢复原始模型参数
                model.load_state_dict(torch.load(os.path.join(self.config.SAVE_DIR, "best_meta_model.pth"),
                                                 map_location=self.config.DEVICE)['model_state_dict'])

        avg_accuracy = np.mean(all_accuracies)
        return avg_accuracy

    def _visualize_noise_results(self, results):
        """可视化噪声实验结果"""
        # 创建结果目录
        vis_dir = os.path.join(self.result_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # 绘制不同噪声类型下的性能曲线
        for noise_type in self.noise_types:
            plt.figure(figsize=(10, 6))

            # 提取性能数据
            snr_levels = sorted(results['standard'][noise_type].keys())
            standard_acc = [results['standard'][noise_type][snr]['accuracy'] for snr in snr_levels]
            meta_acc = [results['meta'][noise_type][snr]['accuracy'] for snr in snr_levels]

            # 绘制性能曲线
            plt.plot(snr_levels, standard_acc, 'o-', label='HRRPGraphNet')
            plt.plot(snr_levels, meta_acc, 's-', label='Meta-HRRPNet')

            # 添加干净数据性能参考线
            plt.axhline(y=results['clean']['standard'], color='blue', linestyle='--', alpha=0.5,
                        label=f'HRRPGraphNet (clean): {results["clean"]["standard"]:.1f}%')
            plt.axhline(y=results['clean']['meta'], color='orange', linestyle='--', alpha=0.5,
                        label=f'Meta-HRRPNet (clean): {results["clean"]["meta"]:.1f}%')

            plt.xlabel('SNR (dB)')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Model Performance under {noise_type.capitalize()} Noise')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()

            # 保存图像
            save_path = os.path.join(vis_dir, f'{noise_type}_performance.png')
            plt.savefig(save_path, dpi=300)

        # 绘制鲁棒性比例图
        plt.figure(figsize=(12, 8))

        # 每种噪声类型一个子图
        for i, noise_type in enumerate(self.noise_types):
            plt.subplot(len(self.noise_types), 1, i + 1)

            # 提取鲁棒性数据
            snr_levels = sorted(results['standard'][noise_type].keys())
            standard_rob = [results['standard'][noise_type][snr]['robustness'] for snr in snr_levels]
            meta_rob = [results['meta'][noise_type][snr]['robustness'] for snr in snr_levels]

            # 绘制鲁棒性曲线
            plt.plot(snr_levels, standard_rob, 'o-', label='HRRPGraphNet')
            plt.plot(snr_levels, meta_rob, 's-', label='Meta-HRRPNet')

            plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='No degradation')

            plt.xlabel('SNR (dB)' if i == len(self.noise_types) - 1 else '')
            plt.ylabel('Robustness (%)')
            plt.title(f'{noise_type.capitalize()} Noise')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(vis_dir, 'robustness_comparison.png')
        plt.savefig(save_path, dpi=300)

        # 绘制鲁棒性提升热力图
        plt.figure(figsize=(10, 6))

        # 准备热力图数据
        robustness_gain = np.zeros((len(self.noise_types), len(self.snr_levels)))

        for i, noise_type in enumerate(self.noise_types):
            for j, snr in enumerate(self.snr_levels):
                meta_rob = results['meta'][noise_type][snr]['robustness']
                standard_rob = results['standard'][noise_type][snr]['robustness']
                gain = meta_rob - standard_rob
                robustness_gain[i, j] = gain

        # 绘制热力图
        ax = sns.heatmap(robustness_gain, annot=True, fmt=".1f", cmap="RdYlGn",
                         xticklabels=self.snr_levels, yticklabels=self.noise_types)

        plt.xlabel('SNR (dB)')
        plt.ylabel('Noise Type')
        plt.title('Robustness Improvement: Meta-HRRPNet vs HRRPGraphNet (%)')
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(vis_dir, 'robustness_gain_heatmap.png')
        plt.savefig(save_path, dpi=300)

        self.logger.info(f"可视化结果已保存至 {vis_dir}")