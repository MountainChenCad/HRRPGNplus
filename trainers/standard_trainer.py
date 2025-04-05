"""
标准训练器 - 用于基线模型的训练与评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class StandardTrainer:
    """
    标准训练器，用于训练和评估原始HRRPGraphNet模型

    参数:
    - model: 要训练的模型
    - train_dataset: 训练数据集
    - val_dataset: 验证数据集
    - test_dataset: 测试数据集
    - config: 配置参数
    - logger: 日志记录器
    """

    def __init__(self, model, train_dataset, val_dataset, test_dataset, config, logger=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config

        # 设置设备
        self.device = config.DEVICE
        self.model.to(self.device)

        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # 设置日志记录器
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # 初始化训练状态
        self.epoch = 0
        self.best_val_acc = 0.0
        self.best_model_path = os.path.join(config.SAVE_DIR, f"best_model_{config.EXP_ID}.pth")

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

        # 创建距离矩阵
        self.distance_matrix = self._generate_distance_matrix(config.FEATURE_DIM).to(self.device)

        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def _generate_distance_matrix(self, N):
        """生成距离矩阵"""
        distance_matrix = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = 1 / (abs(i - j) + 1)
        return distance_matrix

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config.NUM_EPOCHS}")
        for batch_idx, (data, targets) in enumerate(pbar):
            # 将数据移动到设备
            data, targets = data.to(self.device), targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            if hasattr(self.model, 'use_dynamic_graph') and self.model.use_dynamic_graph:
                outputs, _ = self.model(data, self.distance_matrix)
            else:
                outputs = self.model(data, self.distance_matrix)

            # 计算损失
            loss = self.criterion(outputs, targets)
            if self.config.LAMBDA_L2 > 0:
                l2_reg = 0.0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += self.config.LAMBDA_L2 * l2_reg

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            # 更新统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        # 更新历史记录
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(avg_acc)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

        return avg_loss, avg_acc

    def validate(self):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                # 将数据移动到设备
                data, targets = data.to(self.device), targets.to(self.device)

                # 前向传播
                if hasattr(self.model, 'use_dynamic_graph') and self.model.use_dynamic_graph:
                    outputs, _ = self.model(data, self.distance_matrix)
                else:
                    outputs = self.model(data, self.distance_matrix)

                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 统计准确率
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total

        # 更新历史记录
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        # 更新学习率调度器
        self.scheduler.step(avg_loss)

        return avg_loss, avg_acc

    def test(self):
        """在测试集上评估模型性能"""
        self.model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Testing"):
                # 将数据移动到设备
                data, targets = data.to(self.device), targets.to(self.device)

                # 前向传播
                if hasattr(self.model, 'use_dynamic_graph') and self.model.use_dynamic_graph:
                    outputs, _ = self.model(data, self.distance_matrix)
                else:
                    outputs = self.model(data, self.distance_matrix)

                # 记录预测结果
                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # 计算性能指标
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        cm = confusion_matrix(all_targets, all_predictions)

        # 计算每个类别的准确率
        class_accuracies = {}
        for i in range(len(np.unique(all_targets))):
            class_mask = np.array(all_targets) == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(np.array(all_predictions)[class_mask] == i) * 100
                class_accuracies[i] = class_acc

        metrics = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'confusion_matrix': cm,
            'class_accuracies': class_accuracies
        }

        return metrics

    def train(self):
        """训练模型指定轮数"""
        self.logger.info(f"开始训练: {self.config.NUM_EPOCHS} 轮")
        start_time = time.time()

        # 早停计数器
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            self.epoch = epoch

            # 训练一轮
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 记录日志
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(self.best_model_path)
                self.logger.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1

            # 检查是否早停
            if patience_counter >= self.config.PATIENCE:
                self.logger.info(f"早停触发，{self.config.PATIENCE}轮没有提升")
                break

            # 每隔一定轮数保存模型
            if self.config.SAVE_FREQ > 0 and (epoch + 1) % self.config.SAVE_FREQ == 0:
                save_path = os.path.join(
                    self.config.SAVE_DIR,
                    f"model_epoch{epoch + 1}_{self.config.EXP_ID}.pth"
                )
                self.save_checkpoint(save_path)

        # 训练结束
        elapsed_time = time.time() - start_time
        self.logger.info(f"训练完成，耗时: {elapsed_time / 60:.2f}分钟")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.2f}%")

        # 加载最佳模型进行测试
        self.load_checkpoint(self.best_model_path)
        test_metrics = self.test()

        self.logger.info(f"测试准确率: {test_metrics['accuracy']:.2f}%")
        self.logger.info(f"测试精确率: {test_metrics['precision']:.2f}%")
        self.logger.info(f"测试召回率: {test_metrics['recall']:.2f}%")
        self.logger.info(f"测试F1分数: {test_metrics['f1']:.2f}%")

        # 打印每个类别的准确率
        for class_idx, acc in test_metrics['class_accuracies'].items():
            self.logger.info(f"类别 {class_idx} 准确率: {acc:.2f}%")

        return self.history, test_metrics

    def save_checkpoint(self, path):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config.get_config_dict() if hasattr(self.config, 'get_config_dict') else vars(self.config)
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        if not os.path.exists(path):
            self.logger.warning(f"检查点文件不存在: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']

        self.logger.info(f"加载检查点: {path}, Epoch: {self.epoch}, 最佳验证准确率: {self.best_val_acc:.2f}%")
        return True