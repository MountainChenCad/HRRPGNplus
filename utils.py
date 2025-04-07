import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
import os
import pandas as pd
from config import Config
import math


# In utils.py, modify generate_distance_matrix:
def generate_distance_matrix(N):
    """Generate a more sophisticated static adjacency matrix"""
    distance_matrix = torch.zeros(N, N, dtype=torch.float32)

    # Combine both distance and window-based connectivity
    for i in range(N):
        for j in range(N):
            # Distance-based connectivity
            dist_weight = 1.0 / (abs(i - j) + 1)

            # Window-based connectivity (strong local connections)
            window_weight = 1.0 if abs(i - j) <= 5 else 0.0

            # Combined weighting
            distance_matrix[i, j] = 0.7 * dist_weight + 0.3 * window_weight

    return distance_matrix


def prepare_static_adjacency(batch_size, feature_size, device):
    """准备批量的静态邻接矩阵"""
    # 处理极端情况
    if feature_size <= 1:
        print(f"警告: 特征大小 {feature_size} 太小，使用默认大小500")
        feature_size = 500

    static_adj = generate_distance_matrix(feature_size)
    static_adj = static_adj.to(device)
    static_adj = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
    return static_adj


def normalize_adjacency(adj):
    """标准化邻接矩阵 (D^(-1/2) * A * D^(-1/2))"""
    batch_size, N, _ = adj.size()

    # 计算度矩阵
    D = torch.sum(adj, dim=2)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

    # 扩展为对角矩阵
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)

    # 归一化邻接矩阵
    normalized_adj = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)

    return normalized_adj


def compute_metrics(true_labels, pred_labels):
    """计算性能评估指标"""
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )

    # 计算每个类别的准确率
    cm = confusion_matrix(true_labels, pred_labels)
    class_acc = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        'accuracy': acc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'class_accuracy': {i: acc * 100 for i, acc in enumerate(class_acc)},
        'confusion_matrix': cm
    }

    return metrics


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_learning_curve(train_accs, val_accs, title='Learning Curve', save_path=None):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(train_accs) + 1))

    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_shot_curve(shot_sizes, accuracies, methods=None, title='Shot-Accuracy Curve', save_path=None):
    """绘制样本数-精度曲线"""
    plt.figure(figsize=(10, 6))

    if methods is None:
        plt.plot(shot_sizes, accuracies, 'bo-')
    else:
        for i, (method_name, method_accs) in enumerate(zip(methods, accuracies)):
            plt.plot(shot_sizes, method_accs, marker='o', label=method_name)

    plt.title(title)
    plt.xlabel('Number of Shots (K)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    if methods is not None:
        plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_features(features, labels, title='t-SNE Feature Visualization', save_path=None):
    """使用t-SNE可视化特征"""
    # 降维到2D
    tsne = TSNE(n_components=2, random_state=Config.seed)
    features_2d = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[colors[i]], label=f'Class {label}',
            alpha=0.7, edgecolors='k'
        )

    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_dynamic_graph(adjacency, save_path=None):
    """可视化动态图结构"""
    # 可视化邻接矩阵
    plt.figure(figsize=(8, 8))
    sns.heatmap(adjacency.cpu().numpy(), cmap='viridis')
    plt.title('Dynamic Graph Adjacency Matrix')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_attention(data, attention_weights, title='Attention Weights', save_path=None):
    """可视化注意力权重"""
    plt.figure(figsize=(12, 6))

    # 上半部分：HRRP信号
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(data.cpu().numpy().flatten()))
    plt.title('HRRP Signal Magnitude')

    # 下半部分：注意力权重
    plt.subplot(2, 1, 2)
    plt.bar(range(len(attention_weights)), attention_weights.cpu().numpy().flatten())
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def log_metrics(metrics, epoch=None, prefix=''):
    """记录和打印性能指标"""
    log_str = f"{prefix} "
    if epoch is not None:
        log_str += f"Epoch {epoch}: "

    log_str += f"Accuracy: {metrics['accuracy']:.2f}%, "
    log_str += f"Precision: {metrics['precision']:.2f}%, "
    log_str += f"Recall: {metrics['recall']:.2f}%, "
    log_str += f"F1: {metrics['f1']:.2f}%"

    print(log_str)

    # 记录每个类别的准确率
    for class_idx, class_acc in metrics['class_accuracy'].items():
        print(f"{prefix} Class {class_idx} Accuracy: {class_acc:.2f}%")

    return log_str


def save_model(model, save_path):
    """保存模型"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model, load_path):
    """加载模型"""
    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")
    return model


def set_requires_grad(model, requires_grad=True):
    """设置模型是否需要梯度"""
    for param in model.parameters():
        param.requires_grad = requires_grad


def create_experiment_log(config):
    """创建实验日志"""
    log_path = os.path.join(config.log_dir, 'experiment_config.txt')
    with open(log_path, 'w') as f:
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")

    print(f"Experiment configuration logged to {log_path}")


# MAML++ Specific Utility Functions

def adapt_parameters(model, loss, lr=0.01, params=None):
    """Update parameters based on loss, return updated parameter dictionary"""
    if params is None:
        params = {n: p.clone() for n, p in model.named_parameters()}

    # Compute gradients
    grads = torch.autograd.grad(loss, params.values(), create_graph=True, allow_unused=True)

    # Update parameters
    updated_params = {}
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            updated_params[name] = param
        else:
            updated_params[name] = param - lr * grad

    return updated_params


def compute_cosine_annealing_lr(epoch, max_epochs, min_lr, max_lr):
    """Compute learning rate using cosine annealing schedule"""
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs))


def plot_learning_rates(layer_lrs, save_path=None):
    """Plot the learning rates for different layers across steps"""
    plt.figure(figsize=(12, 8))

    for layer_name, step_lrs in layer_lrs.items():
        # Convert tensor values to numpy array
        if isinstance(step_lrs[0], torch.Tensor):
            step_lrs = [lr.item() for lr in step_lrs]
        plt.plot(range(len(step_lrs)), step_lrs, marker='o', label=layer_name)

    plt.title("Per-Layer Learning Rates")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.legend(loc='best')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_learning_rates(layer_lrs, file_path):
    """Save learning rates to CSV file"""
    df = pd.DataFrame(layer_lrs)
    df.to_csv(file_path, index_label='Step')
    print(f"Learning rates saved to {file_path}")


def create_params_dict_from_names(model):
    """Create parameter dictionary with name as key and parameter as value"""
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param
    return params_dict


def clone_params_dict(params_dict):
    """Deep clone a parameter dictionary"""
    return {name: param.clone() for name, param in params_dict.items()}


def log_gradient_norm(model, phase='train'):
    """Log the gradient norm for model parameters"""
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    print(f"{phase} gradient norm: {total_norm}")
    return total_norm


def find_latest_experiment():
    """Find the latest experiment ID"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return None

    experiment_dirs = [d for d in os.listdir(logs_dir) if d.startswith("experiment_")]
    if not experiment_dirs:
        return None

    # Sort by timestamp to get the latest
    latest_exp = sorted(experiment_dirs, reverse=True)[0]
    return latest_exp.replace("experiment_", "")

def get_module_by_name(model, module_name):
    """Get a module from a model by its name"""
    names = module_name.split('.')
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def analyze_model_complexity(model):
    """Analyze model complexity and parameter distribution"""
    # Count parameters by layer
    layer_params = {}
    total_params = 0

    for name, param in model.named_parameters():
        layer_name = name.split('.')[0:2]
        layer_name = '.'.join(layer_name)

        num_params = param.numel()
        total_params += num_params

        if layer_name in layer_params:
            layer_params[layer_name] += num_params
        else:
            layer_params[layer_name] = num_params

    # Print parameter distribution
    print(f"Model has {total_params} total parameters")
    print("\nParameter distribution by layer:")
    for layer_name, num_params in layer_params.items():
        percentage = (num_params / total_params) * 100
        print(f"{layer_name}: {num_params} parameters ({percentage:.2f}%)")

    return layer_params, total_params