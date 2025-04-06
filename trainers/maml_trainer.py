"""
MAML训练器 - 实现元学习训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
import logging
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models.meta_modules import ProtoLoss


class MAMLTrainer:
    """
    MAML训练器，实现元学习训练范式

    参数:
    - model: 元学习模型
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
        self.meta_optimizer = self._get_optimizer(
            config.META_OPTIMIZER,
            self.model.parameters(),
            lr=config.META_LR
        )

        # 创建距离矩阵
        self.distance_matrix = self._generate_distance_matrix(config.FEATURE_DIM).to(self.device)

        # 创建任务采样器
        from data.meta_dataset import TaskSampler, CurriculumTaskSampler
        if config.USE_CURRICULUM:
            self.train_sampler = CurriculumTaskSampler(
                train_dataset, config.N_WAY, config.K_SHOT, config.Q_QUERY,
                num_tasks=config.TASKS_PER_EPOCH, fixed_tasks=True,
                temperature=config.INIT_TEMP, temp_decay=config.TEMP_DECAY
            )
        else:
            self.train_sampler = TaskSampler(
                train_dataset, config.N_WAY, config.K_SHOT, config.Q_QUERY,
                num_tasks=config.TASKS_PER_EPOCH, fixed_tasks=True
            )

        self.val_sampler = TaskSampler(
            val_dataset, config.N_WAY, config.K_SHOT, config.Q_QUERY,
            num_tasks=config.EVAL_TASKS, fixed_tasks=True
        )

        self.test_sampler = TaskSampler(
            test_dataset, config.N_WAY, config.K_SHOT, config.Q_QUERY,
            num_tasks=config.EVAL_TASKS, fixed_tasks=True
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
        self.best_model_path = os.path.join(config.SAVE_DIR, f"best_meta_model_{config.EXP_ID}.pth")

        # 设置辅助损失
        self.proto_loss = ProtoLoss(config.N_WAY, config.K_SHOT)

        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'temperature': [] if config.USE_CURRICULUM else None
        }

    def _get_optimizer(self, optim_type, params, lr):
        """创建优化器"""
        if optim_type == 'Adam':
            return optim.Adam(params, lr=lr)
        elif optim_type == 'SGD':
            return optim.SGD(params, lr=lr, momentum=self.config.META_MOMENTUM)
        else:
            raise ValueError(f"不支持的优化器类型: {optim_type}")

    def _generate_distance_matrix(self, N):
        """Generate normalized distance matrix"""
        # Create raw distance matrix
        distance_matrix = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = 1 / (abs(i - j) + 1)

        # Apply min-max normalization
        min_val = distance_matrix.min()
        max_val = distance_matrix.max()
        if max_val > min_val:
            distance_matrix = (distance_matrix - min_val) / (max_val - min_val)

        return distance_matrix

    def _compute_accuracy(self, logits, targets):
        """计算准确率"""
        _, predicted = torch.max(logits, 1)
        return (predicted == targets).float().mean().item() * 100

    def _fast_adapt(self, support_x, support_y, query_x, query_y, inner_steps, inner_lr):
        """
        快速适应过程（MAML内循环）
        """
        task_losses = []
        task_accuracies = []

        # Validate target values
        if support_y.max() >= self.model.num_classes:
            raise ValueError(
                f"Target labels (max={support_y.max().item()}) exceed model classes ({self.model.num_classes})")
        if support_y.min() < 0:
            raise ValueError(f"Negative target labels found: min={support_y.min().item()}")

        # 克隆模型参数，用于内循环适应
        fast_weights = OrderedDict(
            (name, param.clone()) for (name, param) in self.model.named_parameters()
        )

        # 内循环适应
        for step in range(inner_steps):
            # 前向传播 - 处理(logits, adj_matrix)返回值
            logits, _ = self.model(support_x, self.distance_matrix)

            # 计算损失
            loss = F.cross_entropy(logits, support_y)

            # 计算梯度
            grads = torch.autograd.grad(
                loss,
                [p for p in self.model.parameters() if p.requires_grad],
                create_graph=not self.config.FIRST_ORDER,
                retain_graph=not self.config.FIRST_ORDER
            )

            # 更新权重
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                if param.requires_grad:
                    fast_weights[name] = param - inner_lr * grad

            # 临时用新权重替换模型参数
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = fast_weights[name]

        # 在查询集上评估适应后的模型 - 处理(logits, adj_matrix)返回值
        query_logits, _ = self.model(query_x, self.distance_matrix)

        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = self._compute_accuracy(query_logits, query_y)

        # 收集任务损失和准确率
        task_losses.append(query_loss.item())
        task_accuracies.append(query_acc)

        # 恢复原始模型参数
        original_weights = OrderedDict(
            (name, param.clone()) for (name, param) in self.model.named_parameters()
        )
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = original_weights[name]

        return query_loss, task_losses, task_accuracies

    def meta_train_step(self, meta_batch_size):
        """执行一次元训练步骤"""
        self.model.train()

        # 采样一批任务
        task_batch = self.train_sampler.sample(meta_batch_size)

        # 提取支持集和查询集
        n_way = task_batch['n_way']
        k_shot = task_batch['k_shot']

        support_x = task_batch['support_x']  # [meta_batch_size, n_way*k_shot, ...]
        support_y = task_batch['support_y']  # [meta_batch_size, n_way*k_shot]
        query_x = task_batch['query_x']  # [meta_batch_size, n_way*q_query, ...]
        query_y = task_batch['query_y']  # [meta_batch_size, n_way*q_query]

        # 元优化器梯度清零
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        all_task_losses = []
        all_task_accs = []

        # 对每个任务执行内循环适应
        for i in range(meta_batch_size):
            s_x = support_x[i].to(self.device)
            s_y = support_y[i].to(self.device)
            q_x = query_x[i].to(self.device)
            q_y = query_y[i].to(self.device)

            # 执行快速适应
            task_loss, task_losses, task_accs = self._fast_adapt(
                s_x, s_y, q_x, q_y,
                self.config.INNER_STEPS,
                self.config.INNER_LR
            )

            # 累积元损失
            meta_loss += task_loss
            all_task_losses.extend(task_losses)
            all_task_accs.extend(task_accs)

        # 计算平均元损失
        meta_loss = meta_loss / meta_batch_size

        # 如果启用度量学习损失，添加辅助损失
        if self.config.METRIC_REG > 0:
            # 提取所有任务的特征
            all_support_x = support_x.reshape(-1, *support_x.shape[2:]).to(self.device)
            all_support_y = support_y.reshape(-1).to(self.device)

            # 获取嵌入
            if hasattr(self.model, 'get_embedding'):
                embeddings = self.model.get_embedding(all_support_x, self.distance_matrix)
                # 扁平化特征
                embeddings = embeddings.mean(dim=1)  # [batch_size, feature_dim]

                # 添加原型网络损失
                proto_loss = self.proto_loss(embeddings, all_support_y)
                meta_loss += self.config.METRIC_REG * proto_loss

        # 如果启用正交约束，添加正则化
        if self.config.ORTHO_REG > 0:
            ortho_reg = 0.0
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    # 计算权重矩阵的正交性约束
                    param_flat = param.view(param.size(0), -1)
                    sym = torch.matmul(param_flat, param_flat.t())
                    sym -= torch.eye(param_flat.size(0), device=param.device)
                    ortho_reg += torch.norm(sym)

            meta_loss += self.config.ORTHO_REG * ortho_reg

        # 反向传播和优化
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
        self.meta_optimizer.step()

        # 计算平均任务损失和准确率
        avg_task_loss = np.mean(all_task_losses)
        avg_task_acc = np.mean(all_task_accs)

        return avg_task_loss, avg_task_acc

    def meta_validate(self, meta_batch_size=4):
        """Evaluate the meta-learning model on the validation set"""
        self.model.eval()

        all_task_losses = []
        all_task_accs = []

        # Sample validation tasks
        for i in range(0, self.config.EVAL_TASKS, meta_batch_size):
            # Determine batch size
            curr_batch_size = min(meta_batch_size, self.config.EVAL_TASKS - i)
            if curr_batch_size <= 0:
                break

            task_batch = self.val_sampler.sample(curr_batch_size)

            support_x = task_batch['support_x']
            support_y = task_batch['support_y']
            query_x = task_batch['query_x']
            query_y = task_batch['query_y']

            # Process each task individually
            for j in range(curr_batch_size):
                s_x = support_x[j].to(self.device)
                s_y = support_y[j].to(self.device)
                q_x = query_x[j].to(self.device)
                q_y = query_y[j].to(self.device)

                # Store original parameters
                original_params = OrderedDict(
                    (name, param.clone()) for (name, param) in self.model.named_parameters()
                )

                # Inner loop adaptation - needs gradients
                with torch.enable_grad():
                    fast_weights = OrderedDict(
                        (name, param.clone()) for (name, param) in self.model.named_parameters()
                    )

                    # Inner loop adaptation
                    for step in range(self.config.INNER_STEPS):
                        # 修改为处理 (logits, adj_matrix) 返回值
                        logits, _ = self.model(s_x, self.distance_matrix)
                        loss = F.cross_entropy(logits, s_y)

                        # Calculate gradients
                        grads = torch.autograd.grad(
                            loss,
                            [p for p in self.model.parameters() if p.requires_grad],
                            create_graph=False
                        )

                        # Update weights
                        for (name, param), grad in zip(self.model.named_parameters(), grads):
                            if param.requires_grad:
                                param.data = param - self.config.INNER_LR * grad

                # Evaluate on query set - no gradients needed
                with torch.no_grad():
                    # 修改为处理 (logits, adj_matrix) 返回值
                    query_logits, _ = self.model(q_x, self.distance_matrix)
                    query_loss = F.cross_entropy(query_logits, q_y)
                    query_acc = self._compute_accuracy(query_logits, q_y)

                    all_task_losses.append(query_loss.item())
                    all_task_accs.append(query_acc)

                # Restore original parameters
                for name, param in self.model.named_parameters():
                    param.data = original_params[name]

        # Calculate average task loss and accuracy
        avg_task_loss = np.mean(all_task_losses)
        avg_task_acc = np.mean(all_task_accs)

        return avg_task_loss, avg_task_acc

    def meta_test(self):
        """Comprehensive evaluation on the test set"""
        self.model.eval()

        all_task_losses = []
        all_task_accs = []
        all_predictions = []
        all_targets = []

        # Sample test tasks
        for i in range(0, self.config.EVAL_TASKS, 1):
            task_batch = self.test_sampler.sample(1)

            # Extract support and query sets
            support_x = task_batch['support_x'][0].to(self.device)
            support_y = task_batch['support_y'][0].to(self.device)
            query_x = task_batch['query_x'][0].to(self.device)
            query_y = task_batch['query_y'][0].to(self.device)

            # Store original parameters
            original_params = OrderedDict(
                (name, param.clone()) for (name, param) in self.model.named_parameters()
            )

            # Inner loop adaptation - needs gradients
            with torch.enable_grad():
                # Inner loop adaptation
                for step in range(self.config.INNER_STEPS):
                    # 修改为处理 (logits, adj_matrix) 返回值
                    logits, _ = self.model(support_x, self.distance_matrix)
                    loss = F.cross_entropy(logits, support_y)

                    # Calculate gradients
                    grads = torch.autograd.grad(
                        loss,
                        [p for p in self.model.parameters() if p.requires_grad],
                        create_graph=False
                    )

                    # Update weights
                    for (name, param), grad in zip(self.model.named_parameters(), grads):
                        if param.requires_grad:
                            param.data = param - self.config.INNER_LR * grad

            # Evaluate on query set - no gradients needed
            with torch.no_grad():
                # 修改为处理 (logits, adj_matrix) 返回值
                query_logits, _ = self.model(query_x, self.distance_matrix)
                query_loss = F.cross_entropy(query_logits, query_y)
                query_acc = self._compute_accuracy(query_logits, query_y)

                all_task_losses.append(query_loss.item())
                all_task_accs.append(query_acc)

                # Collect predictions and targets
                _, predicted = torch.max(query_logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(query_y.cpu().numpy())

            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name]

        # Calculate average metrics
        avg_task_loss = np.mean(all_task_losses)
        avg_task_acc = np.mean(all_task_accs)

        # Calculate other performance metrics
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        # Filter valid labels
        valid_indices = (all_targets >= 0)
        all_targets = all_targets[valid_indices]
        all_predictions = all_predictions[valid_indices]

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )

        # Calculate per-class accuracies
        class_accuracies = {}
        for cls in np.unique(all_targets):
            mask = (all_targets == cls)
            class_acc = np.mean(all_predictions[mask] == cls) * 100
            class_accuracies[int(cls)] = class_acc

        metrics = {
            'accuracy': avg_task_acc,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'class_accuracies': class_accuracies,
            'loss': avg_task_loss
        }

        return metrics

    def train(self):
        """执行元学习训练流程"""
        self.logger.info(f"开始元学习训练: {self.config.META_EPOCHS} 轮")
        start_time = time.time()

        # 早停计数器
        patience_counter = 0

        for epoch in range(self.config.META_EPOCHS):
            self.epoch = epoch

            # 更新课程学习温度参数
            if self.config.USE_CURRICULUM and hasattr(self.train_sampler, 'update_temperature'):
                temp = self.train_sampler.update_temperature(epoch)
                self.history['temperature'].append(temp)
                self.logger.info(f"Epoch {epoch + 1}: 温度参数 = {temp:.4f}")

            # 训练阶段
            train_losses = []
            train_accs = []

            # 分批执行元训练步骤
            num_batches = self.config.TASKS_PER_EPOCH // self.config.META_BATCH_SIZE

            pbar = tqdm(range(num_batches), desc=f"Meta Epoch {epoch + 1}/{self.config.META_EPOCHS}")
            for _ in pbar:
                loss, acc = self.meta_train_step(self.config.META_BATCH_SIZE)
                train_losses.append(loss)
                train_accs.append(acc)

                # 更新进度条
                pbar.set_postfix({
                    'loss': np.mean(train_losses),
                    'acc': np.mean(train_accs)
                })

            # 计算平均训练损失和准确率
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)

            # 更新历史记录
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)

            # 验证阶段
            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_loss, val_acc = self.meta_validate()

                # 更新历史记录
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # 记录日志
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.META_EPOCHS} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
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
            else:
                # 仅记录训练日志
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.META_EPOCHS} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%"
                )

            # 每隔一定轮数保存快照
            if self.config.SNAPSHOT_INTERVAL > 0 and (epoch + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                save_path = os.path.join(
                    self.config.SAVE_DIR,
                    f"meta_model_epoch{epoch + 1}_{self.config.EXP_ID}.pth"
                )
                self.save_checkpoint(save_path)

        # 训练结束
        elapsed_time = time.time() - start_time
        self.logger.info(f"元学习训练完成，耗时: {elapsed_time / 60:.2f}分钟")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.2f}%")

        # 加载最佳模型进行测试
        self.load_checkpoint(self.best_model_path)
        test_metrics = self.meta_test()

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
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
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
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']

        self.logger.info(f"加载检查点: {path}, Epoch: {self.epoch}, 最佳验证准确率: {self.best_val_acc:.2f}%")
        return True