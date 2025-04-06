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
    """MAML训练器"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.outer_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.outer_lr,
            weight_decay=Config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def inner_loop(self, support_x, support_y, inner_steps=None, inner_lr=None):
        """MAML内循环更新"""
        inner_steps = inner_steps or Config.inner_steps
        inner_lr = inner_lr or Config.inner_lr

        # 克隆模型进行内循环更新
        updated_model = self.model.clone()
        updated_model.train()

        batch_size, _, seq_len = support_x.size()
        static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)

        # 执行内循环更新
        for _ in range(inner_steps):
            # 前向传播
            logits, _ = updated_model(support_x, static_adj)
            loss = self.criterion(logits, support_y)

            # 更新模型参数
            params = updated_model.adapt_params(loss, lr=inner_lr)
            updated_model.set_params(params)

        return updated_model

    def outer_step(self, tasks, train=True):
        """MAML外循环步骤"""
        outer_loss = 0.0
        accuracies = []

        for task_idx, task in enumerate(tasks):
            support_x, support_y, query_x, query_y = [t.to(self.device) for t in task]

            # 内循环更新
            updated_model = self.inner_loop(support_x, support_y)

            # 在查询集上评估
            batch_size, _, seq_len = query_x.size()
            static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)
            logits, _ = updated_model(query_x, static_adj)
            task_loss = self.criterion(logits, query_y)

            outer_loss += task_loss

            # 计算准确率
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                accuracy = (pred == query_y).float().mean().item()
                accuracies.append(accuracy)

        # 平均所有任务的损失
        outer_loss = outer_loss / len(tasks)

        # 如果是训练模式，执行梯度更新
        if train:
            self.outer_optimizer.zero_grad()
            outer_loss.backward()
            self.outer_optimizer.step()

        return outer_loss.item(), np.mean(accuracies) * 100

    def train_epoch(self, task_generator, num_tasks=None):
        """训练一个epoch"""
        self.model.train()
        num_tasks = num_tasks or Config.task_batch_size
        tasks = []

        # 生成任务批次
        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = task_generator.generate_task()
            tasks.append((support_x, support_y, query_x, query_y))

        # 执行外循环更新
        loss, accuracy = self.outer_step(tasks, train=True)

        return loss, accuracy

    def validate(self, task_generator, num_tasks=100):
        """验证模型"""
        self.model.eval()
        tasks = []

        # 生成验证任务
        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = task_generator.generate_task()
            tasks.append((support_x, support_y, query_x, query_y))

        # 在验证任务上评估
        with torch.no_grad():
            loss, accuracy = self.outer_step(tasks, train=False)

        return loss, accuracy

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
    """测试模型在新类上的性能"""
    num_tasks = num_tasks or Config.num_tasks
    inner_steps = Config.inner_steps
    inner_lr = Config.inner_lr

    # 如果指定了shot，临时修改task generator的k_shot
    original_k_shot = test_task_generator.k_shot
    if shot is not None:
        test_task_generator.k_shot = shot

    model.eval()
    total_accuracy = 0.0
    all_accuracies = []

    for task_idx in tqdm(range(num_tasks), desc="Testing"):
        support_x, support_y, query_x, query_y = test_task_generator.generate_task()
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        # 克隆模型进行内循环更新
        updated_model = model.clone()
        updated_model.train()

        batch_size, _, seq_len = support_x.size()
        static_adj = prepare_static_adjacency(batch_size, seq_len, device)

        # 内循环适应
        for _ in range(inner_steps):
            logits, _ = updated_model(support_x, static_adj)
            loss = F.cross_entropy(logits, support_y)
            params = updated_model.adapt_params(loss, lr=inner_lr)
            updated_model.set_params(params)

        # 在查询集上评估
        updated_model.eval()
        batch_size, _, seq_len = query_x.size()
        static_adj = prepare_static_adjacency(batch_size, seq_len, device)

        with torch.no_grad():
            logits, _ = updated_model(query_x, static_adj)
            pred = torch.argmax(logits, dim=1)

            accuracy = (pred == query_y).float().mean().item()
            all_accuracies.append(accuracy)
            total_accuracy += accuracy

    # 恢复原始k_shot
    if shot is not None:
        test_task_generator.k_shot = original_k_shot

    avg_accuracy = total_accuracy / num_tasks
    std_accuracy = np.std(all_accuracies)
    ci95 = 1.96 * std_accuracy / np.sqrt(num_tasks)

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