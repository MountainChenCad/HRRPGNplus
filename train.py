import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import copy
import math
from config import Config
from models import MDGN
from utils import prepare_static_adjacency, compute_metrics, log_metrics, save_model, load_model


class MAMLPlusPlusTrainer:
    """MAML++ trainer with enhanced features"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.outer_lr,
            weight_decay=Config.weight_decay
        )

        # LR scheduler for cosine annealing
        if Config.use_lr_annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.meta_optimizer,
                T_max=Config.T_max,
                eta_min=Config.min_outer_lr
            )
        else:
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 0  # Track current epoch for annealing

        # Initialize per-layer per-step learning rates as learnable parameters
        self.init_layer_lrs()

    def init_layer_lrs(self):
        """Initialize per-layer per-step learning rates"""
        # Initialize learning rates for different layers
        self.layer_lrs = {}

        if Config.use_per_layer_lr:
            for base_layer in ['feature_extractor', 'graph_generator', 'graph_convs', 'pooling', 'fc']:
                self.layer_lrs[base_layer] = Config.inner_lr

        self.step_lrs = [Config.inner_lr] * Config.inner_steps if Config.use_per_step_lr else [Config.inner_lr]

    def get_inner_lr(self, layer_name, step):
        """Get learning rate for a specific layer and step"""
        if not Config.use_per_layer_lr and not Config.use_per_step_lr:
            return Config.inner_lr

        # Extract base layer name
        if isinstance(layer_name, list):
            # If already a list, join first two elements
            parts = layer_name[:2]
        else:
            # If a string, split and take first two elements
            parts = layer_name.split('.')[:2]

        # Get base layer name (first component of the parameter name)
        base_layer = parts[0] if len(parts) > 0 else "default"

        # Layer-specific learning rate
        layer_lr = self.layer_lrs.get(base_layer, Config.inner_lr) if Config.use_per_layer_lr else Config.inner_lr

        # Step-specific learning rate factor
        step_factor = 1.0
        if Config.use_per_step_lr and step < len(self.step_lrs):
            step_factor = self.step_lrs[step] / self.step_lrs[0]  # Normalize by first step lr

        return layer_lr * step_factor

    def inner_loop(self, support_x, support_y, create_graph=False):
        """Simplified inner loop with multi-step loss"""
        # Clone model to avoid affecting original parameters during inner loop
        updated_model = copy.deepcopy(self.model)
        updated_model.train()

        # Ensure data is on correct device
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)

        batch_size, channels, seq_len = support_x.shape
        static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)

        # Track losses for each step
        step_losses = []
        final_logits = None

        # Inner loop updates
        for step in range(Config.inner_steps):
            # Forward pass
            logits, _ = updated_model(support_x, static_adj)
            loss = self.criterion(logits, support_y)
            step_losses.append(loss)

            if step == Config.inner_steps - 1:
                final_logits = logits

            # Compute gradients
            grads = torch.autograd.grad(
                loss, updated_model.parameters(),
                create_graph=create_graph,
                allow_unused=True
            )

            # Update each parameter with its specific learning rate
            with torch.no_grad():
                for (name, param), grad in zip(updated_model.named_parameters(), grads):
                    if grad is not None:
                        # Get layer-specific learning rate
                        lr = self.get_inner_lr(name, step)
                        # Update parameter
                        param.data.sub_(lr * grad)

        # Compute multi-step loss if enabled
        if Config.use_multi_step_loss:
            # Weighted sum of losses from all steps
            weights = [Config.get_loss_weight(i) for i in range(len(step_losses))]
            multi_step_loss = sum(w * l for w, l in zip(weights, step_losses))
        else:
            # Only use final step loss
            multi_step_loss = step_losses[-1]

        # Create dictionary of updated parameters for reference
        updated_params = {name: param for name, param in updated_model.named_parameters()}

        return updated_model, updated_params, multi_step_loss, final_logits

    def outer_step(self, tasks, train=True):
        """Outer loop optimization with MAML++ enhancements"""
        meta_loss = 0.0
        meta_accuracy = 0.0

        # Determine whether to use second-order gradients based on current epoch
        create_graph = Config.use_second_order and self.epoch >= Config.second_order_start_epoch if train else False

        for task_idx, task in enumerate(tasks):
            support_x, support_y, query_x, query_y = [t.to(self.device) for t in task]

            # Inner loop adaptation - returns updated model
            updated_model, _, support_loss, _ = self.inner_loop(
                support_x,
                support_y,
                create_graph=create_graph
            )

            # Evaluate on query set
            batch_size, channels, seq_len = query_x.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, self.device)

            if train:
                # For training, compute loss and update meta-parameters
                query_logits, _ = updated_model(query_x, static_adj)
                query_loss = self.criterion(query_logits, query_y)

                # Combined loss (support + query)
                task_loss = 0.3 * support_loss.detach() + 0.7 * query_loss
                meta_loss += task_loss

                # Compute accuracy
                pred = torch.argmax(query_logits, dim=1)
                accuracy = (pred == query_y).float().mean().item()
                meta_accuracy += accuracy
            else:
                # For evaluation, no gradient needed
                with torch.no_grad():
                    query_logits, _ = updated_model(query_x, static_adj)
                    query_loss = self.criterion(query_logits, query_y)

                    pred = torch.argmax(query_logits, dim=1)
                    accuracy = (pred == query_y).float().mean().item()
                    meta_accuracy += accuracy
                    meta_loss += query_loss.item()

        # Average loss and accuracy
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = (meta_accuracy / len(tasks)) * 100

        # Perform optimization if in training mode
        if train and meta_loss.requires_grad:
            self.meta_optimizer.zero_grad()
            meta_loss.backward()

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

            # Update model parameters
            self.meta_optimizer.step()

        return meta_loss.item() if train else meta_loss, meta_accuracy

    def train_epoch(self, task_generator, num_tasks=None):
        """Train for one epoch with MAML++ enhancements"""
        self.model.train()
        num_tasks = num_tasks or Config.task_batch_size
        tasks = []

        # Generate task batch
        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = task_generator.generate_task()
            tasks.append((support_x, support_y, query_x, query_y))

        # Process tasks in batch for faster training
        meta_loss, meta_accuracy = self.outer_step(tasks, train=True)

        # Update learning rate with cosine annealing if enabled
        if self.scheduler:
            self.scheduler.step()

        return meta_loss, meta_accuracy

    def validate(self, task_generator, num_tasks=100):
        """Validate model with MAML++ approach"""
        self.model.eval()
        tasks = []

        # Generate validation tasks
        for _ in range(num_tasks):
            try:
                support_x, support_y, query_x, query_y = task_generator.generate_task()
                tasks.append((support_x, support_y, query_x, query_y))
            except Exception as e:
                print(f"Error generating validation task: {e}")
                continue

        if not tasks:
            print("No valid tasks could be generated for validation")
            return 0.0, 0.0

        # Process tasks
        meta_loss, meta_accuracy = self.outer_step(tasks, train=False)

        return meta_loss, meta_accuracy

    def train(self, train_task_generator, val_task_generator, num_epochs=None):
        """Full training procedure with MAML++ enhancements"""
        num_epochs = num_epochs or Config.max_epochs

        best_val_accuracy = 0.0
        best_epoch = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(num_epochs):
            self.epoch = epoch  # Update current epoch for annealing

            # Training phase
            train_loss, train_acc = self.train_epoch(train_task_generator)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation phase
            val_loss, val_acc = self.validate(val_task_generator)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Current learning rate
            current_lr = self.meta_optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                save_model(self.model, os.path.join(Config.save_dir, 'best_model.pth'))
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

            # Early stopping check
            if epoch - best_epoch >= Config.patience:
                print(f"Early stopping triggered after {Config.patience} epochs without improvement")
                break

        # Load best model
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
    """Test model with MAML++ fine-tuning on each task"""
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

            # Create a temporary training instance for this task
            temp_trainer = MAMLPlusPlusTrainer(copy.deepcopy(model), device)

            # Perform inner loop adaptation
            updated_model, _, _, _ = temp_trainer.inner_loop(
                support_x,
                support_y,
                create_graph=False
            )

            # Evaluate on query set
            batch_size, channels, seq_len = query_x.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, device)

            with torch.no_grad():
                query_logits, _ = updated_model(query_x, static_adj)
                pred = torch.argmax(query_logits, dim=1)
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


# Maintain backward compatibility
MAMLTrainer = MAMLPlusPlusTrainer