import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import seaborn as sns
import os
import copy
import math
import time
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from config import Config
from dataset import TaskGenerator
from models import (
    MDGN, CNNModel, LSTMModel, GCNModel, GATModel,
    ProtoNetModel, MatchingNetModel, PCASVM, TemplateMatcher,
    StaticGraphModel, DynamicGraphModel, HybridGraphModel
)
from utils import (
    prepare_static_adjacency, compute_metrics, log_metrics, save_model, load_model,
    visualize_features, visualize_dynamic_graph, visualize_attention, plot_confusion_matrix
)

# Define CVPR-quality color palette
COLORS = ['#0783D5', '#E52119', '#FD751F', '#0E2D88', '#78196D',
          '#C2C121', '#FC837E', '#00A6BC', '#025057', '#7E5505', '#77196C']

# Set global matplotlib parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12

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

        # Add progress bar for task processing
        for task_idx, task in tqdm(enumerate(tasks), total=len(tasks), desc="Processing tasks"):
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

        # Generate task batch with progress bar
        for _ in tqdm(range(num_tasks), desc="Generating training tasks"):
            support_x, support_y, query_x, query_y = task_generator.generate_task()
            tasks.append((support_x, support_y, query_x, query_y))

        # Process tasks in batch for faster training
        meta_loss, meta_accuracy = self.outer_step(tasks, train=True)

        # Update learning rate with cosine annealing if enabled
        if self.scheduler:
            self.scheduler.step()

        return meta_loss, meta_accuracy

    def validate(self, task_generator, num_tasks=4):
        """Validate model with MAML++ approach"""
        self.model.eval()
        tasks = []

        # Generate validation tasks with progress bar
        for _ in tqdm(range(num_tasks), desc="Generating validation tasks"):
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

        # Add tqdm progress bar for epochs
        for epoch in tqdm(range(num_epochs), desc="Training epochs"):
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
    all_f1_scores = []
    all_preds = []
    all_labels = []

    for task_idx in tqdm(range(num_tasks), desc="Testing"):
        try:
            support_x, support_y, query_x, query_y = test_task_generator.generate_task()
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            # 创建一个临时训练实例进行任务适应
            temp_trainer = MAMLPlusPlusTrainer(copy.deepcopy(model), device)

            # 执行内循环适应
            updated_model, _, _, _ = temp_trainer.inner_loop(
                support_x,
                support_y,
                create_graph=False
            )

            # 在查询集上评估
            batch_size, channels, seq_len = query_x.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, device)

            with torch.no_grad():
                # 修改这部分代码以适应不同模型的输出格式
                if hasattr(updated_model, '__class__') and updated_model.__class__.__name__ in ['CNNModel', 'LSTMModel',
                                                                                                'GATModel']:
                    if updated_model.__class__.__name__ == 'GATModel':
                        query_logits, _ = updated_model(query_x)
                    else:
                        # CNN和LSTM模型不需要adjacency matrix作为输入
                        query_logits, _ = updated_model(query_x)
                else:
                    # MDGN和其他需要adjacency matrix的模型
                    query_logits, _ = updated_model(query_x, static_adj)

                pred = torch.argmax(query_logits, dim=1)

                # Collect predictions and labels
                all_preds.extend(pred.detach().cpu().numpy())
                all_labels.extend(query_y.detach().cpu().numpy())

                # Calculate accuracy
                accuracy = (pred == query_y).float().mean().item()
                all_accuracies.append(accuracy)
                total_accuracy += accuracy

                # Calculate F1 score (macro)
                f1 = f1_score(query_y.detach().cpu().numpy(), pred.detach().cpu().numpy(), average='macro')
                all_f1_scores.append(f1)

        except Exception as e:
            print(f"Testing task {task_idx} error: {e}")
            continue

    # Restore original k_shot
    if shot is not None:
        test_task_generator.k_shot = original_k_shot

    if not all_accuracies:
        return 0, 0, [], 0

    # Calculate overall metrics
    avg_accuracy = total_accuracy / len(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    ci95 = 1.96 * std_accuracy / np.sqrt(len(all_accuracies))
    avg_f1 = np.mean(all_f1_scores)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print detailed results
    print(f"Test Accuracy: {avg_accuracy * 100:.2f}% ± {ci95 * 100:.2f}%")
    print(f"Test F1 Score (macro): {avg_f1 * 100:.2f}%")

    # Save confusion matrix
    if not os.path.exists(Config.log_dir):
        os.makedirs(Config.log_dir)
    n_classes = len(np.unique(all_labels))
    class_names = [f"Class {i}" for i in range(n_classes)]
    plot_confusion_matrix(cm, class_names,
                          title=f"{shot}-shot Confusion Matrix",
                          save_path=os.path.join(Config.log_dir, f"confusion_matrix_{shot}shot.png"))

    return avg_accuracy * 100, ci95 * 100, all_accuracies, avg_f1 * 100


def shot_experiment(models, task_generator, device, shot_sizes=None):
    """
    Run experiment with different shot sizes for multiple models

    Args:
        models: Dictionary of model_name -> model pairs or single model
        task_generator: Task generator to sample tasks from
        device: Device to run on
        shot_sizes: List of shot sizes to test

    Returns:
        Dictionary with results for each model
    """
    from utils import compute_metrics, prepare_static_adjacency

    # Handle single model case
    if not isinstance(models, dict):
        models = {'HRRPGraphNet': models}

    # Default shot sizes if not specified
    if shot_sizes is None:
        shot_sizes = [1, 5, 10, 20]

    # Filter to feasible shot sizes
    max_k = task_generator.dataset.get_class_distribution()
    min_samples = min([count for count in max_k.values() if count > 0])
    feasible_shots = [k for k in shot_sizes if k < min_samples - 1]  # Need at least 1 for query

    if not feasible_shots:
        print("Warning: No feasible shot sizes. Using single shot.")
        feasible_shots = [1]

    print(f"Running shot experiment with sizes: {feasible_shots}")

    # Initialize results
    results = {
        'shot_sizes': feasible_shots,
        'models': {}
    }

    for model_name, model in models.items():
        model = model.to(device)
        model_accuracies = []
        model_cis = []
        model_f1s = []

        # For each shot size
        for k in feasible_shots:
            print(f"\nTesting {model_name} with {k}-shot...")

            # Update task generator
            original_k_shot = task_generator.k_shot
            original_q_query = task_generator.q_query
            task_generator.k_shot = k
            task_generator.q_query = min(original_q_query, min_samples - k)

            # Run test
            all_accuracies = []
            all_preds = []
            all_labels = []

            # Test on multiple tasks
            num_tasks = 100  # Config.num_tasks
            for i in range(num_tasks):
                try:
                    # Generate task
                    support_x, support_y, query_x, query_y = task_generator.generate_task()

                    # Move to device
                    support_x = support_x.to(device)
                    support_y = support_y.to(device)
                    query_x = query_x.to(device)
                    query_y = query_y.to(device)

                    # Fine-tune model (MAML-style)
                    if hasattr(model, 'clone'):
                        # MAML models have clone and adapt methods
                        from train import MAMLPlusPlusTrainer
                        temp_trainer = MAMLPlusPlusTrainer(model, device)
                        adapted_model, _, _, _ = temp_trainer.inner_loop(support_x, support_y)
                    else:
                        # Non-MAML models - train on support set directly
                        adapted_model = model

                        # If it's a ProtoNet or MatchingNet, they handle support set directly
                        if model_name in ['ProtoNet', 'MatchingNet']:
                            # Skip adaptation - these models use support set at inference time
                            pass
                        else:
                            # For regular models, fine-tune with basic SGD
                            optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
                            criterion = torch.nn.CrossEntropyLoss()

                            # Simple fine-tuning loop
                            adapted_model.train()
                            for _ in range(5):  # Simple 5-step fine-tuning
                                optimizer.zero_grad()
                                logits, _ = adapted_model(support_x)
                                loss = criterion(logits, support_y)
                                loss.backward()
                                optimizer.step()

                    # Evaluate on query set
                    adapted_model.eval()
                    batch_size, channels, seq_len = query_x.shape
                    static_adj = prepare_static_adjacency(batch_size, seq_len, device)

                    with torch.no_grad():
                        if model_name == 'ProtoNet':
                            # ProtoNet inference
                            support_features = adapted_model(support_x)
                            query_features = adapted_model(query_x)
                            prototypes = adapted_model.compute_prototypes(
                                support_features, support_y, task_generator.n_way)
                            dists = adapted_model.compute_distances(query_features, prototypes)
                            preds = torch.argmin(dists, dim=1)
                        elif model_name == 'MatchingNet':
                            # MatchingNet inference
                            logits, _ = model(query_x, support_set=support_x, support_labels=support_y)

                            # Get predictions
                            _, preds = torch.max(logits, 1)
                        else:
                            # Standard inference
                            logits, _ = adapted_model(query_x, static_adj)
                            preds = torch.argmax(logits, dim=1)

                        # Compute accuracy
                        acc = (preds == query_y).float().mean().item()
                        all_accuracies.append(acc)

                        # Collect predictions and labels for F1 score
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(query_y.cpu().numpy())

                except Exception as e:
                    print(f"Error in task {i}: {e}")
                    continue

            # Compute metrics
            avg_acc = np.mean(all_accuracies) * 100
            std_acc = np.std(all_accuracies) * 100
            ci = 1.96 * std_acc / np.sqrt(len(all_accuracies))

            # Compute F1 score
            from sklearn.metrics import f1_score
            if len(set(all_labels)) > 1:  # Ensure we have at least 2 classes
                f1 = f1_score(all_labels, all_preds, average='macro') * 100
            else:
                f1 = avg_acc  # If only one class, F1 equals accuracy

            print(f"{model_name} {k}-shot accuracy: {avg_acc:.2f}% ± {ci:.2f}%, F1: {f1:.2f}%")

            model_accuracies.append(avg_acc)
            model_cis.append(ci)
            model_f1s.append(f1)

            # Restore original task generator settings
            task_generator.k_shot = original_k_shot
            task_generator.q_query = original_q_query

        # Store results for this model
        results['models'][model_name] = {
            'accuracies': model_accuracies,
            'confidence_intervals': model_cis,
            'f1_scores': model_f1s
        }

    return results


def ablation_study_lambda(model, test_task_generator, device, lambda_values=None):
    """Lambda mixing coefficient ablation experiment"""
    if lambda_values is None:
        lambda_values = Config.ablation['lambda_values']

    results = []
    ci_results = []
    f1_results = []
    original_lambda = Config.lambda_mix

    # Create results directory
    results_dir = os.path.join(Config.log_dir, 'lambda_ablation')
    os.makedirs(results_dir, exist_ok=True)

    # Save results in DataFrame
    results_data = []

    for lambda_val in lambda_values:
        print(f"\nTesting with lambda={lambda_val}:")
        Config.lambda_mix = lambda_val
        accuracy, ci, _, f1 = test_model(model, test_task_generator, device, num_tasks=100)
        results.append(accuracy)
        ci_results.append(ci)
        f1_results.append(f1)
        print(f"Lambda={lambda_val} Accuracy: {accuracy:.2f}% ± {ci:.2f}%, F1: {f1:.2f}%")

        # Add to results data
        results_data.append({
            'Lambda': lambda_val,
            'Accuracy': accuracy,
            'CI': ci,
            'F1': f1
        })

    # Restore original lambda value
    Config.lambda_mix = original_lambda

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(results_dir, 'lambda_ablation_results.csv'), index=False)

    # Plot results with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(lambda_values, results, yerr=ci_results, fmt='o-', capsize=5, label='Accuracy')
    plt.plot(lambda_values, f1_results, 's--', label='F1 Score')
    plt.title('Performance vs Lambda Mixing Coefficient')
    plt.xlabel('Lambda (Static Graph Weight)')
    plt.ylabel('Performance (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'lambda_ablation_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return lambda_values, results, ci_results, f1_results


def shot_experiment(dataset, device, baseline_models=None, shot_sizes=None):
    """
    Compare models across different shot sizes

    Args:
        dataset: Test dataset
        device: Device to run models on
        baseline_models: List of baseline models to compare
        shot_sizes: List of shot sizes to test

    Returns:
        Dictionary with results for each model and shot size
    """
    # Default shot sizes if not specified
    if shot_sizes is None:
        shot_sizes = [1, 5, 10, 20]

    # Filter to feasible shot sizes
    max_k = dataset.get_class_distribution()
    min_samples = min([count for count in max_k.values() if count > 0])
    feasible_shots = [k for k in shot_sizes if k < min_samples - 1]  # Need at least 1 for query

    if not feasible_shots:
        print("Warning: No feasible shot sizes. Using single shot.")
        feasible_shots = [1]

    # Initialize results
    results = {
        'shot_sizes': feasible_shots,
        'models': {}
    }

    # Run comparison for each shot size
    for k_shot in feasible_shots:
        print(f"\nComparing models with {k_shot}-shot...")

        # Run baseline comparison for this shot value
        comparison_results = compare_with_baselines(
            dataset, device, shot=k_shot,
            baseline_models=baseline_models,
            num_tasks=50  # Use fewer tasks for efficiency
        )

        # Extract results
        for _, row in comparison_results.iterrows():
            model_name = row['model']

            # Initialize model entry if not exists
            if model_name not in results['models']:
                results['models'][model_name] = {
                    'accuracies': [],
                    'confidence_intervals': [],
                    'f1_scores': []
                }

            # Parse accuracy and confidence interval
            acc_text = row['accuracy']
            acc_parts = acc_text.split('±')
            acc_value = float(acc_parts[0].strip().replace('%', ''))
            ci_value = float(acc_parts[1].strip().replace('%', '')) if len(acc_parts) > 1 else 0

            # Parse F1 score
            f1_value = float(row['f1_score'].replace('%', ''))

            # Add to results
            results['models'][model_name]['accuracies'].append(acc_value)
            results['models'][model_name]['confidence_intervals'].append(ci_value)
            results['models'][model_name]['f1_scores'].append(f1_value)

    # Generate plots
    # 1. Accuracy plot
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)

    model_names = list(results['models'].keys())
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, model_name in enumerate(model_names):
        model_data = results['models'][model_name]
        color = COLORS[i % len(COLORS)]
        marker = markers[i % len(markers)]

        ax.errorbar(
            feasible_shots,
            model_data['accuracies'],
            yerr=model_data['confidence_intervals'],
            fmt=f'-{marker}',
            linewidth=2.5,
            markersize=8,
            capsize=5,
            label=model_name,
            color=color
        )

    # Styling
    ax.set_title('Model Accuracy vs Number of Shots', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Use integer ticks for shot sizes
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(Config.log_dir, 'baseline_comparison', 'shot_comparison', 'accuracy_vs_shots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. F1 score plot
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)

    for i, model_name in enumerate(model_names):
        model_data = results['models'][model_name]
        color = COLORS[i % len(COLORS)]
        marker = markers[i % len(markers)]

        ax.plot(
            feasible_shots,
            model_data['f1_scores'],
            f'-{marker}',
            linewidth=2.5,
            markersize=8,
            label=model_name,
            color=color
        )

    # Styling
    ax.set_title('Model F1 Score vs Number of Shots', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Use integer ticks for shot sizes
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(Config.log_dir, 'baseline_comparison', 'shot_comparison', 'f1_vs_shots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return results


def compare_models_across_shots(dataset, device, baseline_models=None, shot_sizes=None):
    """
    Compare models across different shot sizes

    Args:
        dataset: Test dataset
        device: Device to run models on
        baseline_models: List of baseline models to compare
        shot_sizes: List of shot sizes to test

    Returns:
        Dictionary with results for each model and shot size
    """
    # Default shot sizes if not specified
    if shot_sizes is None:
        shot_sizes = [1, 5, 10, 20]

    # Filter to feasible shot sizes
    class_dist = dataset.get_class_distribution()
    min_samples = min([count for count in class_dist.values() if count > 0])
    feasible_shots = [k for k in shot_sizes if k < min_samples - 1]  # Need at least 1 for query

    if not feasible_shots:
        print("Warning: No feasible shot sizes. Using single shot.")
        feasible_shots = [1]

    # Initialize results
    results = {
        'shot_sizes': feasible_shots,
        'models': {}
    }

    # Run comparison for each shot size
    for k_shot in feasible_shots:
        print(f"\nComparing models with {k_shot}-shot...")

        # Run baseline comparison for this shot value
        comparison_results = compare_with_baselines(
            dataset, device, shot=k_shot,
            baseline_models=baseline_models,
            num_tasks=50  # Use fewer tasks for efficiency
        )

        # Extract results
        for _, row in comparison_results.iterrows():
            model_name = row['model']

            # Initialize model entry if not exists
            if model_name not in results['models']:
                results['models'][model_name] = {
                    'accuracies': [],
                    'confidence_intervals': [],
                    'f1_scores': []
                }

            # Parse accuracy and confidence interval
            acc_text = row['accuracy']
            acc_parts = acc_text.split('±')
            acc_value = float(acc_parts[0].strip().replace('%', ''))
            ci_value = float(acc_parts[1].strip().replace('%', '')) if len(acc_parts) > 1 else 0

            # Parse F1 score
            f1_value = float(row['f1_score'].replace('%', ''))

            # Add to results
            results['models'][model_name]['accuracies'].append(acc_value)
            results['models'][model_name]['confidence_intervals'].append(ci_value)
            results['models'][model_name]['f1_scores'].append(f1_value)

    # Generate plots
    # 1. Accuracy plot
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)

    model_names = list(results['models'].keys())
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, model_name in enumerate(model_names):
        model_data = results['models'][model_name]
        color = COLORS[i % len(COLORS)]
        marker = markers[i % len(markers)]

        ax.errorbar(
            feasible_shots,
            model_data['accuracies'],
            yerr=model_data['confidence_intervals'],
            fmt=f'-{marker}',
            linewidth=2.5,
            markersize=8,
            capsize=5,
            label=model_name,
            color=color
        )

    # Styling
    ax.set_title('Model Accuracy vs Number of Shots', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Use integer ticks for shot sizes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(Config.log_dir, 'baseline_comparison', 'shot_comparison', 'accuracy_vs_shots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. F1 score plot
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)

    for i, model_name in enumerate(model_names):
        model_data = results['models'][model_name]
        color = COLORS[i % len(COLORS)]
        marker = markers[i % len(markers)]

        ax.plot(
            feasible_shots,
            model_data['f1_scores'],
            f'-{marker}',
            linewidth=2.5,
            markersize=8,
            label=model_name,
            color=color
        )

    # Styling
    ax.set_title('Model F1 Score vs Number of Shots', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Use integer ticks for shot sizes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(Config.log_dir, 'baseline_comparison', 'shot_comparison', 'f1_vs_shots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return results

def ablation_study_dynamic_graph(model, test_task_generator, device):
    """Dynamic vs static graph ablation experiment"""
    results = []
    ci_results = []
    f1_results = []
    labels = ["Static Graph", "Dynamic Graph", "Hybrid Graph"]

    # Create results directory
    results_dir = os.path.join(Config.log_dir, 'graph_structure_ablation')
    os.makedirs(results_dir, exist_ok=True)

    # Save original config
    original_use_dynamic = Config.use_dynamic_graph
    original_lambda = Config.lambda_mix

    # Save results in DataFrame
    results_data = []

    # Test static graph
    print("\nTesting with static graph:")
    Config.use_dynamic_graph = False
    accuracy_static, ci_static, _, f1_static = test_model(model, test_task_generator, device, num_tasks=100)
    results.append(accuracy_static)
    ci_results.append(ci_static)
    f1_results.append(f1_static)
    print(f"Static Graph Accuracy: {accuracy_static:.2f}% ± {ci_static:.2f}%, F1: {f1_static:.2f}%")
    results_data.append({
        'Graph Type': 'Static',
        'Accuracy': accuracy_static,
        'CI': ci_static,
        'F1': f1_static
    })

    # Test dynamic graph (pure)
    print("\nTesting with pure dynamic graph:")
    Config.use_dynamic_graph = True
    Config.lambda_mix = 0.0  # Pure dynamic
    accuracy_dynamic, ci_dynamic, _, f1_dynamic = test_model(model, test_task_generator, device, num_tasks=100)
    results.append(accuracy_dynamic)
    ci_results.append(ci_dynamic)
    f1_results.append(f1_dynamic)
    print(f"Dynamic Graph Accuracy: {accuracy_dynamic:.2f}% ± {ci_dynamic:.2f}%, F1: {f1_dynamic:.2f}%")
    results_data.append({
        'Graph Type': 'Dynamic',
        'Accuracy': accuracy_dynamic,
        'CI': ci_dynamic,
        'F1': f1_dynamic
    })

    # Test hybrid graph
    print("\nTesting with hybrid graph:")
    Config.use_dynamic_graph = True
    Config.lambda_mix = 0.5  # Equal mix
    accuracy_hybrid, ci_hybrid, _, f1_hybrid = test_model(model, test_task_generator, device, num_tasks=100)
    results.append(accuracy_hybrid)
    ci_results.append(ci_hybrid)
    f1_results.append(f1_hybrid)
    print(f"Hybrid Graph Accuracy: {accuracy_hybrid:.2f}% ± {ci_hybrid:.2f}%, F1: {f1_hybrid:.2f}%")
    results_data.append({
        'Graph Type': 'Hybrid',
        'Accuracy': accuracy_hybrid,
        'CI': ci_hybrid,
        'F1': f1_hybrid
    })

    # Restore original settings
    Config.use_dynamic_graph = original_use_dynamic
    Config.lambda_mix = original_lambda

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(results_dir, 'graph_structure_results.csv'), index=False)

    # Plot results with error bars
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(labels))

    # Plot accuracy bars
    plt.bar(x_pos - 0.2, results, width=0.4, yerr=ci_results, capsize=5, label='Accuracy', color='blue', alpha=0.7)

    # Plot F1 bars
    plt.bar(x_pos + 0.2, f1_results, width=0.4, label='F1 Score', color='green', alpha=0.7)

    plt.xticks(x_pos, labels)
    plt.title('Performance Comparison: Graph Structure Types')
    plt.ylabel('Performance (%)')
    plt.grid(axis='y')
    plt.legend()

    plt.savefig(os.path.join(results_dir, 'graph_structure_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return labels, results, ci_results, f1_results


def ablation_study_gnn_architecture(model, test_task_generator, device):
    """GNN architecture components ablation study"""
    # Create results directory
    results_dir = os.path.join(Config.log_dir, 'gnn_architecture_ablation')
    os.makedirs(results_dir, exist_ok=True)

    # Save original config
    original_heads = Config.attention_heads
    original_layers = Config.graph_conv_layers

    # Test different attention heads
    head_values = [2, 4, 8]
    head_results = []
    head_ci = []
    head_f1 = []

    print("\nTesting different attention head configurations:")
    for heads in head_values:
        # Update config
        Config.attention_heads = heads

        # Reinitialize model with new config
        temp_model = MDGN(num_classes=model.encoder.fc.out_features).to(device)
        # Load pre-trained weights where possible
        try:
            temp_model.load_state_dict(model.state_dict())
        except:
            print(f"Warning: Could not load pre-trained weights for heads={heads} model")

        # Test model
        print(f"\nTesting with {heads} attention heads:")
        accuracy, ci, _, f1 = test_model(temp_model, test_task_generator, device, num_tasks=50)
        head_results.append(accuracy)
        head_ci.append(ci)
        head_f1.append(f1)
        print(f"{heads} Attention Heads: Accuracy: {accuracy:.2f}% ± {ci:.2f}%, F1: {f1:.2f}%")

    # Test different graph convolution layers
    layer_values = [1, 2, 3, 4]
    layer_results = []
    layer_ci = []
    layer_f1 = []

    print("\nTesting different graph convolution layer configurations:")
    for layers in layer_values:
        # Update config
        Config.graph_conv_layers = layers

        # Reinitialize model with new config
        temp_model = MDGN(num_classes=model.encoder.fc.out_features).to(device)
        # Note: Cannot load pre-trained weights due to architecture difference

        # Test model
        print(f"\nTesting with {layers} graph convolution layers:")
        accuracy, ci, _, f1 = test_model(temp_model, test_task_generator, device, num_tasks=50)
        layer_results.append(accuracy)
        layer_ci.append(ci)
        layer_f1.append(f1)
        print(f"{layers} Graph Conv Layers: Accuracy: {accuracy:.2f}% ± {ci:.2f}%, F1: {f1:.2f}%")

    # Restore original config
    Config.attention_heads = original_heads
    Config.graph_conv_layers = original_layers

    # Save results
    heads_df = pd.DataFrame({
        'Heads': head_values,
        'Accuracy': head_results,
        'CI': head_ci,
        'F1': head_f1
    })

    layers_df = pd.DataFrame({
        'Layers': layer_values,
        'Accuracy': layer_results,
        'CI': layer_ci,
        'F1': layer_f1
    })

    heads_df.to_csv(os.path.join(results_dir, 'attention_heads_results.csv'), index=False)
    layers_df.to_csv(os.path.join(results_dir, 'conv_layers_results.csv'), index=False)

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot attention heads results
    plt.subplot(1, 2, 1)
    plt.errorbar(head_values, head_results, yerr=head_ci, fmt='o-', capsize=5, label='Accuracy')
    plt.plot(head_values, head_f1, 's--', label='F1 Score')
    plt.title('Performance vs Attention Heads')
    plt.xlabel('Number of Attention Heads')
    plt.ylabel('Performance (%)')
    plt.grid(True)
    plt.legend()

    # Plot graph conv layers results
    plt.subplot(1, 2, 2)
    plt.errorbar(layer_values, layer_results, yerr=layer_ci, fmt='o-', capsize=5, label='Accuracy')
    plt.plot(layer_values, layer_f1, 's--', label='F1 Score')
    plt.title('Performance vs Graph Conv Layers')
    plt.xlabel('Number of Graph Conv Layers')
    plt.ylabel('Performance (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gnn_architecture_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'head_values': head_values,
        'head_results': head_results,
        'head_ci': head_ci,
        'head_f1': head_f1,
        'layer_values': layer_values,
        'layer_results': layer_results,
        'layer_ci': layer_ci,
        'layer_f1': layer_f1
    }


def noise_robustness_experiment(model, test_task_generator, device, noise_levels=None):
    """Noise robustness experiment"""
    if noise_levels is None:
        noise_levels = Config.noise_levels

    results = []
    ci_results = []
    f1_results = []

    # Create results directory
    results_dir = os.path.join(Config.log_dir, 'noise_robustness')
    os.makedirs(results_dir, exist_ok=True)

    # Save original settings
    original_augmentation = Config.augmentation
    original_noise_levels = Config.noise_levels.copy() if hasattr(Config.noise_levels, 'copy') else Config.noise_levels

    # Enable augmentation
    Config.augmentation = True

    # Save results in DataFrame
    results_data = []

    for snr in noise_levels:
        print(f"\nTesting with SNR={snr}dB:")
        Config.noise_levels = [snr]  # Set single noise level
        accuracy, ci, _, f1 = test_model(model, test_task_generator, device, num_tasks=100)
        results.append(accuracy)
        ci_results.append(ci)
        f1_results.append(f1)
        print(f"SNR={snr}dB Accuracy: {accuracy:.2f}% ± {ci:.2f}%, F1: {f1:.2f}%")

        # Add to results data
        results_data.append({
            'SNR': snr,
            'Accuracy': accuracy,
            'CI': ci,
            'F1': f1
        })

    # Restore original settings
    Config.augmentation = original_augmentation
    Config.noise_levels = original_noise_levels

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(results_dir, 'noise_robustness_results.csv'), index=False)

    # Plot results with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(noise_levels, results, yerr=ci_results, fmt='o-', capsize=5, label='Accuracy')
    plt.plot(noise_levels, f1_results, 's--', label='F1 Score')
    plt.title('Performance vs Signal-to-Noise Ratio (SNR)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Performance (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'noise_robustness_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return noise_levels, results, ci_results, f1_results


def compare_with_baselines(dataset, device, shot=5, baseline_models=None, num_tasks=100):
    """
    Compare HRRPGraphNet with baseline methods

    Args:
        dataset: Test dataset
        device: Device to run models on
        shot: K-shot setting
        baseline_models: List of baseline models to compare
        num_tasks: Number of tasks to evaluate

    Returns:
        DataFrame with comparison results
    """
    from models import (CNNModel, LSTMModel, GCNModel, GATModel,
                        ProtoNetModel, MatchingNetModel, MDGN,
                        PCASVM, TemplateMatcher)
    from utils import compute_metrics, prepare_static_adjacency, create_comparison_table
    from torch.optim import Adam
    import torch.nn as nn
    import pandas as pd
    import numpy as np

    # Set up model constructors
    model_constructors = {
        'CNN': CNNModel,
        'LSTM': LSTMModel,
        'GCN': GCNModel,
        'GAT': GATModel,
        'ProtoNet': ProtoNetModel,
        'MatchingNet': MatchingNetModel,
        'PCA+SVM': PCASVM,
        'Template': TemplateMatcher,
        'HRRPGraphNet': MDGN
    }

    # Determine which models to compare based on provided list or config
    if baseline_models is None:
        # Use default baselines from config
        traditional_models = Config.traditional_baselines['methods'] if Config.traditional_baselines['enabled'] else []
        dl_models = Config.dl_baselines['methods'] if Config.dl_baselines['enabled'] else []
        fsl_models = Config.fsl_baselines['methods'] if Config.fsl_baselines['enabled'] else []

        baseline_models = traditional_models + dl_models + fsl_models

    # Make sure HRRPGraphNet is included for comparison
    if 'HRRPGraphNet' not in baseline_models:
        baseline_models = ['HRRPGraphNet'] + baseline_models

    print(f"Comparing against baseline models: {baseline_models}")

    # Set up task generator
    from dataset import TaskGenerator
    n_way = Config.n_way
    k_shot = shot
    q_query = 15  # Fixed query size
    task_generator = TaskGenerator(dataset, n_way=n_way, k_shot=k_shot, q_query=q_query)

    # Results storage
    results = {}

    # Traditional ML baselines evaluation
    traditional_models = ['PCA+SVM', 'Template']
    for model_name in [m for m in baseline_models if m in traditional_models]:
        print(f"\nEvaluating {model_name}...")

        all_accuracies = []
        all_preds = []
        all_labels = []

        for task_idx in range(num_tasks):
            try:
                # Generate task
                support_x, support_y, query_x, query_y = task_generator.generate_task()

                # Convert tensors to numpy arrays
                support_x_np = support_x.numpy()
                support_y_np = support_y.numpy()
                query_x_np = query_x.numpy()
                query_y_np = query_y.numpy()

                # Reshape data
                support_x_np = support_x_np.reshape(support_x_np.shape[0], -1)  # Flatten
                query_x_np = query_x_np.reshape(query_x_np.shape[0], -1)  # Flatten

                # Initialize model
                if model_name == 'PCA+SVM':
                    model = PCASVM(n_components=min(30, support_x_np.shape[0] - 1))
                else:  # Template matching
                    model = TemplateMatcher(metric='correlation')

                # Train on support set
                model.fit(support_x_np, support_y_np)

                # Predict
                preds = model.predict(query_x_np)

                # Calculate accuracy
                acc = np.mean(preds == query_y_np)
                all_accuracies.append(acc)

                # Store predictions and labels for metrics
                all_preds.extend(preds)
                all_labels.extend(query_y_np)

            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue

        # Calculate metrics
        if all_preds:
            metrics = compute_metrics(all_labels, all_preds)
            avg_acc = np.mean(all_accuracies) * 100
            std_acc = np.std(all_accuracies) * 100
            ci = 1.96 * std_acc / np.sqrt(len(all_accuracies))

            results[model_name] = {
                'accuracy': avg_acc,
                'confidence_interval': ci,
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }

            print(f"{model_name} accuracy: {avg_acc:.2f}% ± {ci:.2f}%, F1: {metrics['f1']:.2f}%")

    # Deep learning baselines evaluation
    dl_models = ['CNN', 'LSTM', 'GCN', 'GAT']
    for model_name in [m for m in baseline_models if m in dl_models]:
        print(f"\nEvaluating {model_name}...")

        all_accuracies = []
        all_preds = []
        all_labels = []

        for task_idx in range(num_tasks):
            try:
                # Generate task
                support_x, support_y, query_x, query_y = task_generator.generate_task()

                # Move to device
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                query_x = query_x.to(device)
                query_y = query_y.to(device)

                # Initialize model
                model = model_constructors[model_name](num_classes=n_way)
                model = model.to(device)

                # Train on support set
                model.train()
                optimizer = Adam(model.parameters(), lr=0.01)
                criterion = nn.CrossEntropyLoss()

                # Simple training loop
                batch_size, channels, seq_len = support_x.shape
                static_adj = prepare_static_adjacency(batch_size, seq_len, device)

                # 50 epochs of fine-tuning
                for _ in range(50):
                    optimizer.zero_grad()
                    logits, _ = model(support_x, static_adj)
                    loss = criterion(logits, support_y)
                    loss.backward()
                    optimizer.step()

                # Evaluate on query set
                model.eval()
                batch_size, channels, seq_len = query_x.shape
                static_adj = prepare_static_adjacency(batch_size, seq_len, device)

                with torch.no_grad():
                    logits, _ = model(query_x, static_adj)
                    preds = torch.argmax(logits, dim=1)

                    # Calculate accuracy
                    acc = (preds == query_y).float().mean().item()
                    all_accuracies.append(acc)

                    # Store predictions and labels for metrics
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(query_y.cpu().numpy())

            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue

        # Calculate metrics
        if all_preds:
            metrics = compute_metrics(all_labels, all_preds)
            avg_acc = np.mean(all_accuracies) * 100
            std_acc = np.std(all_accuracies) * 100
            ci = 1.96 * std_acc / np.sqrt(len(all_accuracies))

            results[model_name] = {
                'accuracy': avg_acc,
                'confidence_interval': ci,
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }

            print(f"{model_name} accuracy: {avg_acc:.2f}% ± {ci:.2f}%, F1: {metrics['f1']:.2f}%")

    # Few-shot learning baselines evaluation
    fsl_models = ['ProtoNet', 'MatchingNet']
    for model_name in [m for m in baseline_models if m in fsl_models]:
        print(f"\nEvaluating {model_name}...")

        all_accuracies = []
        all_preds = []
        all_labels = []

        # Initialize model
        model = model_constructors[model_name]()
        model = model.to(device)

        # For FSL models, we don't need to train separately for each task
        # Instead, train once on multiple support examples
        if model_name in ['ProtoNet', 'MatchingNet']:
            # Meta-training phase
            model.train()
            optimizer = Adam(model.parameters(), lr=0.001)

            # Meta-train on a batch of tasks
            meta_train_epochs = 100
            meta_batch_size = 4

            print(f"Meta-training {model_name} for {meta_train_epochs} epochs...")
            for epoch in range(meta_train_epochs):
                meta_train_loss = 0.0
                meta_train_acc = 0.0

                for _ in range(meta_batch_size):
                    # Generate task
                    support_x, support_y, query_x, query_y = task_generator.generate_task()

                    # Move to device
                    support_x = support_x.to(device)
                    support_y = support_y.to(device)
                    query_x = query_x.to(device)
                    query_y = query_y.to(device)

                    # Forward pass
                    optimizer.zero_grad()

                    if model_name == 'ProtoNet':
                        # ProtoNet training
                        # Extract features
                        support_features = model(support_x)
                        query_features = model(query_x)

                        # Compute prototypes
                        prototypes = model.compute_prototypes(support_features, support_y, n_way)

                        # Compute distances
                        dists = model.compute_distances(query_features, prototypes)

                        # Compute loss (negative log probability)
                        log_p_y = -dists
                        loss = nn.CrossEntropyLoss()(log_p_y, query_y)

                        # Get predictions
                        _, preds = torch.min(dists, 1)

                    elif model_name == 'MatchingNet':

                        # MatchingNet training

                        logits, _ = model(query_x, support_set=support_x, support_labels=support_y)

                        # Compute loss

                        loss = nn.CrossEntropyLoss()(logits, query_y)

                        # Get predictions

                        _, preds = torch.max(logits, 1)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Compute accuracy
                    acc = (preds == query_y).float().mean().item()

                    meta_train_loss += loss.item()
                    meta_train_acc += acc

                # Log progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{meta_train_epochs} - Loss: {meta_train_loss / meta_batch_size:.4f}, "
                          f"Acc: {meta_train_acc / meta_batch_size * 100:.2f}%")

        # Evaluate model on test tasks
        model.eval()
        for task_idx in range(num_tasks):
            try:
                # Generate task
                support_x, support_y, query_x, query_y = task_generator.generate_task()

                # Move to device
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                query_x = query_x.to(device)
                query_y = query_y.to(device)

                with torch.no_grad():
                    if model_name == 'ProtoNet':
                        # ProtoNet inference
                        support_features = model(support_x)
                        query_features = model(query_x)

                        # Compute prototypes
                        prototypes = model.compute_prototypes(support_features, support_y, n_way)

                        # Compute distances
                        dists = model.compute_distances(query_features, prototypes)

                        # Get predictions
                        _, preds = torch.min(dists, 1)

                    # And later in the evaluation section:

                    elif model_name == 'MatchingNet':

                        # MatchingNet inference

                        logits, _ = model(query_x, support_set=support_x, support_labels=support_y)

                        # Get predictions

                        _, preds = torch.max(logits, 1)

                    # Calculate accuracy
                    acc = (preds == query_y).float().mean().item()
                    all_accuracies.append(acc)

                    # Store predictions and labels for metrics
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(query_y.cpu().numpy())

            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue

        # Calculate metrics
        if all_preds:
            metrics = compute_metrics(all_labels, all_preds)
            avg_acc = np.mean(all_accuracies) * 100
            std_acc = np.std(all_accuracies) * 100
            ci = 1.96 * std_acc / np.sqrt(len(all_accuracies))

            results[model_name] = {
                'accuracy': avg_acc,
                'confidence_interval': ci,
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }

            print(f"{model_name} accuracy: {avg_acc:.2f}% ± {ci:.2f}%, F1: {metrics['f1']:.2f}%")

    # HRRPGraphNet evaluation (MAML-style)
    if 'HRRPGraphNet' in baseline_models:
        print("\nEvaluating HRRPGraphNet...")

        # Load the latest trained model if available
        from utils import find_latest_experiment
        model = MDGN(num_classes=n_way)
        model = model.to(device)

        # Try to load from checkpoint if available
        try:
            latest_exp = find_latest_experiment()
            if latest_exp:
                model_path = f"checkpoints/experiment_{latest_exp}/best_model.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))
                    print(f"Loaded HRRPGraphNet model from {model_path}")
        except Exception as e:
            print(f"Error loading saved model: {e}")
            print("Using a freshly initialized HRRPGraphNet model instead.")

        # Evaluate MAML-style
        from train import MAMLPlusPlusTrainer

        all_accuracies = []
        all_preds = []
        all_labels = []

        for task_idx in range(num_tasks):
            try:
                # Generate task
                support_x, support_y, query_x, query_y = task_generator.generate_task()

                # Move to device
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                query_x = query_x.to(device)
                query_y = query_y.to(device)

                # Fine-tune model on support set (MAML inner loop)
                temp_trainer = MAMLPlusPlusTrainer(model, device)
                adapted_model, _, _, _ = temp_trainer.inner_loop(support_x, support_y)

                # Test on query set
                batch_size, channels, seq_len = query_x.shape
                static_adj = prepare_static_adjacency(batch_size, seq_len, device)

                with torch.no_grad():
                    logits, _ = adapted_model(query_x, static_adj)
                    preds = torch.argmax(logits, dim=1)

                    # Calculate accuracy
                    acc = (preds == query_y).float().mean().item()
                    all_accuracies.append(acc)

                    # Store predictions and labels for metrics
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(query_y.cpu().numpy())

            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue

        # Calculate metrics
        if all_preds:
            metrics = compute_metrics(all_labels, all_preds)
            avg_acc = np.mean(all_accuracies) * 100
            std_acc = np.std(all_accuracies) * 100
            ci = 1.96 * std_acc / np.sqrt(len(all_accuracies))

            results['HRRPGraphNet'] = {
                'accuracy': avg_acc,
                'confidence_interval': ci,
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }

            print(f"HRRPGraphNet accuracy: {avg_acc:.2f}% ± {ci:.2f}%, F1: {metrics['f1']:.2f}%")

    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            'model': model_name,
            'accuracy': f"{results[model_name]['accuracy']:.2f}% ± {results[model_name]['confidence_interval']:.2f}%",
            'f1_score': f"{results[model_name]['f1']:.2f}%",
            'precision': f"{results[model_name]['precision']:.2f}%",
            'recall': f"{results[model_name]['recall']:.2f}%"
        }
        for model_name in results.keys()
    ])

    print("\nBaseline Comparison Results:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def visualize_model_interpretability(model, test_task_generator, device):
    """Generate visualizations for model interpretability"""
    # Create visualization directory
    vis_dir = os.path.join(Config.log_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Generate a few tasks for visualization
    num_vis_tasks = 5
    visualization_tasks = []

    print("\nGenerating tasks for visualization:")
    for _ in range(num_vis_tasks):
        try:
            task = test_task_generator.generate_task()
            visualization_tasks.append(task)
        except Exception as e:
            print(f"Error generating visualization task: {e}")
            continue

    if not visualization_tasks:
        print("No valid tasks could be generated for visualization")
        return

    for task_idx, (support_x, support_y, query_x, query_y) in enumerate(visualization_tasks):
        print(f"\nVisualizing task {task_idx + 1}:")
        task_dir = os.path.join(vis_dir, f'task_{task_idx + 1}')
        os.makedirs(task_dir, exist_ok=True)

        # Copy data to device
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        # Fine-tune model on support set
        temp_trainer = MAMLPlusPlusTrainer(copy.deepcopy(model), device)
        updated_model, _, _, _ = temp_trainer.inner_loop(support_x, support_y, create_graph=False)

        # 1. Visualize attention weights
        print("  Visualizing attention weights...")
        batch_size, channels, seq_len = query_x.shape

        for i in range(min(5, batch_size)):  # Visualize first 5 samples
            sample_x = query_x[i:i + 1]
            sample_y = query_y[i:i + 1]

            # 创建匹配单个样本批次大小的邻接矩阵
            sample_static_adj = prepare_static_adjacency(1, seq_len, device)

            # Forward pass with attention extraction
            with torch.no_grad():
                # 检查模型类型
                if hasattr(updated_model, '__class__') and updated_model.__class__.__name__ in ['CNNModel', 'LSTMModel', 'GATModel']:
                    if updated_model.__class__.__name__ == 'GATModel':
                        outputs = updated_model(sample_x, return_features=True, extract_attention=True)
                        logits, features, adj_matrix, attention_weights = outputs
                    else:
                        # CNN和LSTM模型
                        outputs = updated_model(sample_x, return_features=True, extract_attention=True)
                        logits, features, adj_matrix, attention_weights = outputs
                else:
                    # MDGN和其他需要adjacency matrix的模型
                    outputs = updated_model(sample_x, sample_static_adj, return_features=True, extract_attention=True)
                    logits, features, adj_matrix, attention_weights = outputs

            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]
            true_label = sample_y.detach().cpu().numpy()[0]

            # Visualize attention weights
            visualize_attention(
                sample_x.detach().cpu(),
                attention_weights.detach().cpu(),
                title=f'Sample {i + 1}: True: {true_label}, Pred: {pred}',
                save_path=os.path.join(task_dir, f'attention_sample_{i + 1}.png')
            )

        # 2. Visualize dynamic graph structure
        print("  Visualizing dynamic graph structure...")
        # Get a random sample from query set
        sample_idx = np.random.randint(0, batch_size)
        sample_x = query_x[sample_idx:sample_idx + 1]

        # 创建匹配单个样本批次大小的邻接矩阵
        sample_static_adj = prepare_static_adjacency(1, seq_len, device)

        # Forward pass to get adjacency matrix
        with torch.no_grad():
            _, adj_matrix = updated_model(sample_x, sample_static_adj)

        # Visualize adjacency matrix
        visualize_dynamic_graph(
            adj_matrix[0].detach().cpu(),  # First sample in batch
            save_path=os.path.join(task_dir, 'dynamic_graph_structure.png')
        )

        # 3. Visualize feature space
        print("  Visualizing feature space...")
        # Extract features for all query samples
        batch_features = []
        batch_labels = []

        with torch.no_grad():
            # 为每个样本单独处理
            for i in range(batch_size):
                sample_x = query_x[i:i + 1]
                sample_y = query_y[i:i + 1]

                # 创建匹配单个样本批次大小的邻接矩阵
                sample_static_adj = prepare_static_adjacency(1, seq_len, device)

                # Forward pass to extract features
                _, features = updated_model(sample_x, sample_static_adj, return_features=True)[0:2]

                batch_features.append(features.detach().cpu().numpy())
                batch_labels.append(sample_y.detach().cpu().numpy()[0])

        # Stack features and visualize
        batch_features = np.vstack(batch_features)
        batch_labels = np.array(batch_labels)

        visualize_features(
            batch_features,
            batch_labels,
            title='t-SNE Feature Visualization',
            save_path=os.path.join(task_dir, 'feature_space.png')
        )

    print(f"\nAll visualizations saved to {vis_dir}")


def computational_complexity_analysis(test_task_generator, device):
    """Analyze computational complexity of the model"""
    # Create results directory
    results_dir = os.path.join(Config.log_dir, 'computational_complexity')
    os.makedirs(results_dir, exist_ok=True)

    # Generate a sample task for timing
    support_x, support_y, query_x, query_y = test_task_generator.generate_task()
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    batch_size, channels, seq_len = query_x.shape
    static_adj = prepare_static_adjacency(batch_size, seq_len, device)

    # Define models to compare
    num_classes = Config.test_n_way
    models = {
        'HRRPGraphNet++': MDGN(num_classes=num_classes).to(device),
        'CNN': CNNModel(num_classes=num_classes).to(device),
        'LSTM': LSTMModel(num_classes=num_classes).to(device),
        'GCN': GCNModel(num_classes=num_classes).to(device),
        'GAT': GATModel(num_classes=num_classes).to(device)
    }

    # Count parameters
    param_counts = {}
    for name, model_instance in models.items():
        param_count = sum(p.numel() for p in model_instance.parameters())
        param_counts[name] = param_count
        print(f"{name} parameter count: {param_count:,}")

    # Measure inference time
    inference_times = {}
    num_runs = 100

    for name, model_instance in models.items():
        model_instance.eval()

        # Warm-up
        with torch.no_grad():
            if name in ['HRRPGraphNet++', 'GCN']:
                _ = model_instance(query_x, static_adj)
            else:
                _ = model_instance(query_x)

        # Timing runs
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                if name in ['HRRPGraphNet++', 'GCN']:
                    _ = model_instance(query_x, static_adj)
                else:
                    _ = model_instance(query_x)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()

        avg_time = (end_time - start_time) * 1000 / num_runs  # ms
        inference_times[name] = avg_time
        print(f"{name} average inference time: {avg_time:.2f} ms")

    # Measure training (adaptation) time
    adaptation_times = {}

    for name, model_instance in models.items():
        # Only measure for models with MAML adaptation
        if name == 'HRRPGraphNet++':
            # Clone model for adaptation
            clone_model = copy.deepcopy(model_instance)

            # Warm-up
            temp_trainer = MAMLPlusPlusTrainer(clone_model, device)
            _ = temp_trainer.inner_loop(support_x, support_y, create_graph=False)

            # Timing runs
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            for _ in range(5):  # Fewer runs due to higher computational cost
                clone_model = copy.deepcopy(model_instance)
                temp_trainer = MAMLPlusPlusTrainer(clone_model, device)
                _ = temp_trainer.inner_loop(support_x, support_y, create_graph=False)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            avg_time = (end_time - start_time) * 1000 / 5  # ms
            adaptation_times[name] = avg_time
            print(f"{name} average adaptation time: {avg_time:.2f} ms")

    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': list(param_counts.keys()),
        'Parameters': list(param_counts.values()),
        'Inference Time (ms)': [inference_times.get(model, float('nan')) for model in param_counts.keys()],
        'Adaptation Time (ms)': [adaptation_times.get(model, float('nan')) for model in param_counts.keys()]
    })

    results_df.to_csv(os.path.join(results_dir, 'computational_complexity.csv'), index=False)

    # Create bar charts
    plt.figure(figsize=(15, 10))

    # Parameter count subplot
    plt.subplot(2, 2, 1)
    models_list = list(param_counts.keys())
    params_list = list(param_counts.values())

    plt.bar(models_list, params_list)
    plt.title('Model Parameter Count')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    # Inference time subplot
    plt.subplot(2, 2, 2)
    inf_times = [inference_times.get(model, 0) for model in models_list]

    plt.bar(models_list, inf_times)
    plt.title('Inference Time')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    # Adaptation time subplot (only for applicable models)
    plt.subplot(2, 2, 3)
    adapt_models = [model for model in models_list if model in adaptation_times]
    adapt_times = [adaptation_times[model] for model in adapt_models]

    plt.bar(adapt_models, adapt_times)
    plt.title('Adaptation Time (MAML)')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    # Log scale parameter count
    plt.subplot(2, 2, 4)
    plt.bar(models_list, params_list)
    plt.title('Model Parameter Count (Log Scale)')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'computational_complexity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return results_df


def ablation_study_meta_learning(model, test_task_generator, device):
    """元学习组件消融实验"""
    # 创建结果目录
    results_dir = os.path.join(Config.log_dir, 'meta_learning_ablation')
    os.makedirs(results_dir, exist_ok=True)

    # 保存原始配置
    original_per_layer_lr = Config.use_per_layer_lr
    original_per_step_lr = Config.use_per_step_lr
    original_multi_step_loss = Config.use_multi_step_loss
    original_second_order = Config.use_second_order

    # 保存结果数据
    results_data = []

    # 测试所有组合
    test_combinations = []

    # 根据配置生成测试组合
    if Config.meta_learning_ablation['enabled']:
        for per_layer in Config.meta_learning_ablation['per_layer_lr_enabled']:
            for per_step in Config.meta_learning_ablation['per_step_lr_enabled']:
                for multi_step in Config.meta_learning_ablation['multi_step_loss_enabled']:
                    for second_order in Config.meta_learning_ablation['second_order_enabled']:
                        test_combinations.append({
                            'per_layer_lr': per_layer,
                            'per_step_lr': per_step,
                            'multi_step_loss': multi_step,
                            'second_order': second_order,
                            'name': f"PL:{per_layer}_PS:{per_step}_MS:{multi_step}_SO:{second_order}"
                        })

    # 如果组合太多，可以选择一些关键组合
    if len(test_combinations) > 8:
        # 标准MAML (所有增强都关闭)
        test_combinations = [
            {'per_layer_lr': False, 'per_step_lr': False, 'multi_step_loss': False, 'second_order': False,
             'name': 'Standard MAML'},
            # 完整MAML++ (所有增强都开启)
            {'per_layer_lr': True, 'per_step_lr': True, 'multi_step_loss': True, 'second_order': True,
             'name': 'Full MAML++'},
            # 单一组件测试
            {'per_layer_lr': True, 'per_step_lr': False, 'multi_step_loss': False, 'second_order': False,
             'name': 'Only Per-Layer LR'},
            {'per_layer_lr': False, 'per_step_lr': True, 'multi_step_loss': False, 'second_order': False,
             'name': 'Only Per-Step LR'},
            {'per_layer_lr': False, 'per_step_lr': False, 'multi_step_loss': True, 'second_order': False,
             'name': 'Only Multi-Step Loss'},
            {'per_layer_lr': False, 'per_step_lr': False, 'multi_step_loss': False, 'second_order': True,
             'name': 'Only Second-Order'}
        ]

    # 对每种组合进行测试
    performance_results = []
    for config_set in test_combinations:
        print(f"\n测试元学习配置: {config_set['name']}")

        # 应用配置
        Config.use_per_layer_lr = config_set['per_layer_lr']
        Config.use_per_step_lr = config_set['per_step_lr']
        Config.use_multi_step_loss = config_set['multi_step_loss']
        Config.use_second_order = config_set['second_order']

        # 测试模型
        accuracy, ci, _, f1 = test_model(model, test_task_generator, device, num_tasks=50)

        # 记录结果
        result = {
            'Name': config_set['name'],
            'Per_Layer_LR': config_set['per_layer_lr'],
            'Per_Step_LR': config_set['per_step_lr'],
            'Multi_Step_Loss': config_set['multi_step_loss'],
            'Second_Order': config_set['second_order'],
            'Accuracy': accuracy,
            'CI': ci,
            'F1': f1
        }

        results_data.append(result)
        performance_results.append((config_set['name'], accuracy, ci, f1))

        print(f"配置 {config_set['name']} 精度: {accuracy:.2f}% ± {ci:.2f}%, F1: {f1:.2f}%")

    # 还原原始配置
    Config.use_per_layer_lr = original_per_layer_lr
    Config.use_per_step_lr = original_per_step_lr
    Config.use_multi_step_loss = original_multi_step_loss
    Config.use_second_order = original_second_order

    # 保存结果到CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(results_dir, 'meta_learning_ablation_results.csv'), index=False)

    # 创建可视化
    plt.figure(figsize=(12, 8))

    # 结果排序
    names = [r[0] for r in performance_results]
    accs = [r[1] for r in performance_results]
    cis = [r[2] for r in performance_results]
    f1s = [r[3] for r in performance_results]

    # 按精度降序排序
    sorted_indices = np.argsort(accs)[::-1]
    names = [names[i] for i in sorted_indices]
    accs = [accs[i] for i in sorted_indices]
    cis = [cis[i] for i in sorted_indices]
    f1s = [f1s[i] for i in sorted_indices]

    x_pos = np.arange(len(names))

    # 绘制精度条形图
    plt.bar(x_pos - 0.2, accs, width=0.4, yerr=cis, capsize=5, label='Accuracy', color='blue', alpha=0.7)

    # 绘制F1分数条形图
    plt.bar(x_pos + 0.2, f1s, width=0.4, label='F1 Score', color='green', alpha=0.7)

    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.title('Meta-Learning Component Ablation Study')
    plt.ylabel('Performance (%)')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, 'meta_learning_ablation_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建热力图以显示组件重要性
    if len(results_data) >= 4:
        plt.figure(figsize=(10, 8))

        # 准备热力图数据
        components = ['Per_Layer_LR', 'Per_Step_LR', 'Multi_Step_Loss', 'Second_Order']
        heatmap_data = np.zeros((2, len(components)))

        # 计算每个组件开启/关闭的平均性能
        for component_idx, component in enumerate(components):
            enabled_accs = [r['Accuracy'] for r in results_data if r[component]]
            disabled_accs = [r['Accuracy'] for r in results_data if not r[component]]

            heatmap_data[0, component_idx] = np.mean(disabled_accs) if disabled_accs else 0  # OFF
            heatmap_data[1, component_idx] = np.mean(enabled_accs) if enabled_accs else 0  # ON

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                    xticklabels=[c.replace('_', ' ') for c in components],
                    yticklabels=['OFF', 'ON'])

        plt.title('Component Importance Heatmap (Average Accuracy)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'component_importance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return results_data

def computational_complexity_analysis(task_generator, device, save_dir=None):
    """
    Perform comprehensive computational complexity analysis on different models

    Args:
        task_generator: Task generator to create sample tasks
        device: Device to run models on
        save_dir: Directory to save results

    Returns:
        Dictionary of complexity metrics for different models
    """
    from utils import analyze_model_complexity, calculate_model_memory, timing_measurement
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import torch
    from thop import profile
    from models import MDGN, CNNModel, LSTMModel, GCNModel, GATModel

    # Create save directory if provided
    if save_dir is None:
        save_dir = os.path.join(Config.log_dir, 'computational_complexity')
    os.makedirs(save_dir, exist_ok=True)

    # Create models to analyze
    models = {
        'HRRPGraphNet': MDGN(num_classes=Config.test_n_way),
        'CNN': CNNModel(num_classes=Config.test_n_way),
        'LSTM': LSTMModel(num_classes=Config.test_n_way),
        'GCN': GCNModel(num_classes=Config.test_n_way),
        'GAT': GATModel(num_classes=Config.test_n_way)
    }

    # Initialize results dictionary
    results = {
        'model_name': [],
        'total_params': [],
        'trainable_params': [],
        'memory_mb': [],
        'inference_time_ms': [],
        'flops': [],
        'macs': []
    }

    # Generate a sample task for analysis
    support_x, support_y, query_x, query_y = task_generator.generate_task()
    sample_input = query_x[0:1].to(device)  # Use a single sample for timing

    # Compute statistics for each model
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        model = model.to(device)

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Memory usage
        memory_mb = calculate_model_memory(model)

        # Inference time
        model.eval()
        inference_time = timing_measurement(model, sample_input, device, num_runs=100)

        # FLOPs and MACs calculation using thop
        try:
            batch_size, channels, seq_len = sample_input.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, device)
            flops, params = profile(model, inputs=(sample_input, static_adj), verbose=False)
            macs = flops / 2  # Approximately half of FLOPs are MACs
        except Exception as e:
            print(f"Error calculating FLOPs for {name}: {e}")
            flops = 0
            macs = 0

        # Add to results
        results['model_name'].append(name)
        results['total_params'].append(total_params)
        results['trainable_params'].append(trainable_params)
        results['memory_mb'].append(memory_mb)
        results['inference_time_ms'].append(inference_time)
        results['flops'].append(flops)
        results['macs'].append(macs)

        # Save detailed model analysis
        layer_params, _ = analyze_model_complexity(model)
        with open(os.path.join(save_dir, f'{name}_complexity.txt'), 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Memory Usage: {memory_mb:.2f} MB\n")
            f.write(f"Inference Time: {inference_time:.2f} ms\n")
            f.write(f"FLOPs: {flops:,}\n")
            f.write(f"MACs: {macs:,}\n\n")
            f.write("Parameter distribution by layer:\n")
            for layer_name, num_params in layer_params.items():
                percentage = (num_params / total_params) * 100
                f.write(f"{layer_name}: {num_params:,} parameters ({percentage:.2f}%)\n")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, 'model_complexity_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Complexity results saved to {csv_path}")

    # Create visualization charts
    # 1. Parameter count comparison
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)
    x = np.arange(len(models))
    width = 0.35

    # Use COLORS from global definitions
    color1 = '#0783D5'  # COLORS[0]
    color2 = '#E52119'  # COLORS[1]

    ax.bar(x - width / 2, df['total_params'], width, label='Total Params', color=color1)
    ax.bar(x + width / 2, df['trainable_params'], width, label='Trainable Params', color=color2)

    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
    ax.set_title('Model Parameter Count Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Inference time comparison
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)

    ax.bar(df['model_name'], df['inference_time_ms'], color='#FD751F')

    ax.set_title('Model Inference Time Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Add time labels on top of bars
    for i, v in enumerate(df['inference_time_ms']):
        ax.text(i, v + 0.5, f"{v:.2f}ms", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Memory usage comparison
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.subplot(111)

    ax.bar(df['model_name'], df['memory_mb'], color='#0E2D88')

    ax.set_title('Model Memory Usage Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Add memory labels on top of bars
    for i, v in enumerate(df['memory_mb']):
        ax.text(i, v + 0.5, f"{v:.2f}MB", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_usage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return results