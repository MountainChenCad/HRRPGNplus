import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import copy
from config import Config
from models import MDGN
from utils import prepare_static_adjacency, compute_metrics, log_metrics, save_model, load_model


class MAMLTrainer:
    """Simplified trainer that maintains gradient flow"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.outer_lr,
            weight_decay=Config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def inner_loop(self, support_x, support_y):
        """简化的内循环，使用标准优化器"""
        print(f"Internal shape check - support_x: {support_x.shape}")

        # 克隆模型
        updated_model = copy.deepcopy(self.model)
        updated_model.train()

        # 确保数据在正确设备上
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        # 创建内循环优化器
        inner_optimizer = torch.optim.SGD(
            updated_model.parameters(),
            lr=Config.inner_lr
        )

        batch_size, channels, seq_len = support_x.shape
        static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)

        # 内循环更新
        for step in range(Config.inner_steps):
            inner_optimizer.zero_grad()
            logits, _ = updated_model(support_x, static_adj)
            loss = self.criterion(logits, support_y)
            loss.backward()
            inner_optimizer.step()

            if step == 0:  # 只打印第一步的损失
                print(f"  Inner step 1 loss: {loss.item():.4f}")

        return updated_model

    def outer_step(self, tasks, train=True):
        """Outer loop step with direct training on support set"""
        outer_loss = 0.0
        accuracies = []

        for task_idx, task in enumerate(tasks):
            support_x, support_y, query_x, query_y = [t.to(self.device) for t in task]

            if train:
                # Direct training on support set
                batch_size, channels, seq_len = support_x.shape
                static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)

                self.meta_optimizer.zero_grad()
                logits, _ = self.model(support_x, static_adj)
                loss = self.criterion(logits, support_y)
                loss.backward()
                self.meta_optimizer.step()

                # Evaluate on query set (no gradient)
                with torch.no_grad():
                    batch_size, channels, seq_len = query_x.shape
                    static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)
                    logits, _ = self.model(query_x, static_adj)
                    accuracy = (torch.argmax(logits, dim=1) == query_y).float().mean().item()
                    accuracies.append(accuracy)
            else:
                # For evaluation, keep inner loop adaptation
                updated_model = self.inner_loop(support_x, support_y)

                with torch.no_grad():
                    batch_size, channels, seq_len = query_x.shape
                    static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)
                    logits, _ = updated_model(query_x, static_adj)
                    task_loss = self.criterion(logits, query_y)
                    outer_loss += task_loss.item()

                    pred = torch.argmax(logits, dim=1)
                    accuracy = (pred == query_y).float().mean().item()
                    accuracies.append(accuracy)

        return outer_loss / len(tasks) if not train else 0.0, np.mean(accuracies) * 100

    def train_epoch(self, task_generator, num_tasks=None):
        """Train for one epoch"""
        self.model.train()
        num_tasks = num_tasks or Config.task_batch_size
        tasks = []

        # Generate task batch
        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = task_generator.generate_task()
            tasks.append((support_x, support_y, query_x, query_y))

        # Training loop
        total_loss = 0
        total_accuracy = 0

        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(tasks):
            # Move to device
            support_x, support_y = support_x.to(self.device), support_y.to(self.device)
            query_x, query_y = query_x.to(self.device), query_y.to(self.device)

            # Forward pass on support set
            self.meta_optimizer.zero_grad()

            # Process support set
            support_batch_size, channels, seq_len = support_x.shape
            support_static_adj = prepare_static_adjacency(support_batch_size, seq_len, self.device)

            # Train directly on support set
            support_logits, _ = self.model(support_x, support_static_adj)
            support_loss = self.criterion(support_logits, support_y)

            # Process query set to measure performance
            query_batch_size, _, seq_len = query_x.shape
            query_static_adj = prepare_static_adjacency(query_batch_size, seq_len, self.device)

            query_logits, _ = self.model(query_x, query_static_adj)
            query_loss = self.criterion(query_logits, query_y)

            # Combined loss with emphasis on query performance
            loss = 0.3 * support_loss + 0.7 * query_loss

            # Backward and optimize
            loss.backward()
            self.meta_optimizer.step()

            # Calculate accuracy
            pred = torch.argmax(query_logits, dim=1)
            accuracy = (pred == query_y).float().mean().item()

            total_loss += loss.item()
            total_accuracy += accuracy

        return total_loss / num_tasks, (total_accuracy / num_tasks) * 100

    def validate(self, task_generator, num_tasks=100):
        """Validate model"""
        self.model.eval()
        tasks = []

        # Generate validation tasks
        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = task_generator.generate_task()
            tasks.append((support_x, support_y, query_x, query_y))

        # Evaluation loop
        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in tasks:
                # Move to device
                support_x, support_y = support_x.to(self.device), support_y.to(self.device)
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)

                # Process query set
                batch_size, _, seq_len = query_x.shape
                static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)

                logits, _ = self.model(query_x, static_adj)
                loss = self.criterion(logits, query_y)

                # Calculate accuracy
                pred = torch.argmax(logits, dim=1)
                accuracy = (pred == query_y).float().mean().item()

                total_loss += loss.item()
                total_accuracy += accuracy

        return total_loss / num_tasks, (total_accuracy / num_tasks) * 100

    def train(self, train_task_generator, val_task_generator, num_epochs=None):
        """完整训练流程"""
        num_epochs = num_epochs or Config.max_epochs

        best_val_accuracy = 0.0
        best_epoch = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_task_generator)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # 验证阶段
            val_loss, val_acc = self.validate(val_task_generator)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                save_model(self.model, os.path.join(Config.save_dir, 'best_model.pth'))
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

            # 早停检查
            if epoch - best_epoch >= Config.patience:
                print(f"Early stopping triggered after {Config.patience} epochs without improvement")
                break

        # 加载最佳模型
        load_model(self.model, os.path.join(Config.save_dir, 'best_model.pth'))

        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy
        }


def test_model(model, test_task_generator, device, num_tasks=None, shot=None):
    """Test model with fine-tuning on each task"""
    num_tasks = num_tasks or Config.num_tasks

    # Save original k_shot and set new value if specified
    original_k_shot = test_task_generator.k_shot
    if shot is not None:
        test_task_generator.k_shot = shot

    model.eval()
    total_accuracy = 0.0
    all_accuracies = []

    for task_idx in tqdm(range(num_tasks), desc="Testing"):
        try:
            support_x, support_y, query_x, query_y = test_task_generator.generate_task()
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            # Fine-tune a copy of the model on the support set
            adapted_model = copy.deepcopy(model)
            adapted_model.train()

            optimizer = torch.optim.SGD(adapted_model.parameters(), lr=Config.inner_lr)
            criterion = nn.CrossEntropyLoss()

            batch_size, channels, seq_len = support_x.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, device)

            # Fine-tuning steps
            for _ in range(Config.inner_steps):
                optimizer.zero_grad()
                logits, _ = adapted_model(support_x, static_adj)
                loss = criterion(logits, support_y)
                loss.backward()
                optimizer.step()

            # Evaluate on query set
            adapted_model.eval()
            batch_size, channels, seq_len = query_x.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, device)

            with torch.no_grad():
                logits, _ = adapted_model(query_x, static_adj)
                pred = torch.argmax(logits, dim=1)
                accuracy = (pred == query_y).float().mean().item()
                all_accuracies.append(accuracy)
                total_accuracy += accuracy

        except Exception as e:
            print(f"Testing task {task_idx} error: {e}")
            continue

    # Restore original k_shot
    if shot is not None:
        test_task_generator.k_shot = original_k_shot

    if not all_accuracies:
        return 0, 0, []

    avg_accuracy = total_accuracy / len(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    ci95 = 1.96 * std_accuracy / np.sqrt(len(all_accuracies))

    print(f"Test Accuracy: {avg_accuracy * 100:.2f}% ± {ci95 * 100:.2f}%")

    return avg_accuracy * 100, ci95 * 100, all_accuracies


def shot_experiment(model, test_task_generator, device, shot_sizes=None):
    """不同shot下的性能实验"""
    if shot_sizes is None:
        shot_sizes = [1, 2, 3, 5, 10, 20]

    results = []

    for shot in shot_sizes:
        print(f"\nTesting with {shot}-shot:")
        accuracy, ci, _ = test_model(model, test_task_generator, device, num_tasks=100, shot=shot)
        results.append(accuracy)
        print(f"{shot}-shot Accuracy: {accuracy:.2f}% ± {ci:.2f}%")

    return shot_sizes, results


def ablation_study_lambda(model, test_task_generator, device, lambda_values=None):
    """lambda混合比例的消融实验"""
    if lambda_values is None:
        lambda_values = Config.ablation['lambda_values']

    results = []
    original_lambda = Config.lambda_mix

    for lambda_val in lambda_values:
        print(f"\nTesting with lambda={lambda_val}:")
        Config.lambda_mix = lambda_val
        accuracy, ci, _ = test_model(model, test_task_generator, device, num_tasks=100)
        results.append(accuracy)
        print(f"Lambda={lambda_val} Accuracy: {accuracy:.2f}% ± {ci:.2f}%")

    # 恢复原始lambda值
    Config.lambda_mix = original_lambda

    return lambda_values, results


def ablation_study_dynamic_graph(model, test_task_generator, device):
    """动态图vs静态图消融实验"""
    results = []
    labels = ["Static Graph", "Dynamic Graph"]

    # 测试静态图
    original_use_dynamic = Config.use_dynamic_graph
    Config.use_dynamic_graph = False
    accuracy_static, ci_static, _ = test_model(model, test_task_generator, device, num_tasks=100)
    results.append(accuracy_static)
    print(f"Static Graph Accuracy: {accuracy_static:.2f}% ± {ci_static:.2f}%")

    # 测试动态图
    Config.use_dynamic_graph = True
    accuracy_dynamic, ci_dynamic, _ = test_model(model, test_task_generator, device, num_tasks=100)
    results.append(accuracy_dynamic)
    print(f"Dynamic Graph Accuracy: {accuracy_dynamic:.2f}% ± {ci_dynamic:.2f}%")

    # 恢复原始设置
    Config.use_dynamic_graph = original_use_dynamic

    return labels, results


def noise_robustness_experiment(model, test_task_generator, device, noise_levels=None):
    """噪声鲁棒性实验"""
    if noise_levels is None:
        noise_levels = Config.noise_levels

    results = []
    # 保存原始噪声设置
    original_augmentation = Config.augmentation
    Config.augmentation = True

    for snr in noise_levels:
        print(f"\nTesting with SNR={snr}dB:")
        Config.noise_levels = [snr]  # 设置为单一噪声水平
        accuracy, ci, _ = test_model(model, test_task_generator, device, num_tasks=100)
        results.append(accuracy)
        print(f"SNR={snr}dB Accuracy: {accuracy:.2f}% ± {ci:.2f}%")

    # 恢复原始设置
    Config.augmentation = original_augmentation

    return noise_levels, results