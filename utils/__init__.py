"""
评估指标模块 - 提供各种模型评估指标
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    auc, classification_report
)


def calculate_metrics(y_true, y_pred, y_score=None, average='weighted'):
    """
    计算多种评估指标

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_score: 预测概率或分数 (可选，用于计算AUC等)
    - average: 多类别指标的平均方式 ('micro', 'macro', 'weighted', 'samples')

    返回:
    - 包含多种评估指标的字典
    """
    # 确保输入为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_score is not None and isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()

    # 基本分类指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0) * 100,
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0) * 100,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    # 计算各类别的准确率
    class_accuracies = {}
    classes = np.unique(np.concatenate([y_true, y_pred]))
    for cls in classes:
        mask = (y_true == cls)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == cls) * 100
            class_accuracies[int(cls)] = class_acc

    metrics['class_accuracies'] = class_accuracies

    # 如果提供了概率分数，计算AUC等指标
    if y_score is not None:
        try:
            if y_score.shape[1] > 2:  # 多类别情况
                # 对于多类别，计算每个类别的ROC AUC，然后取平均
                if average == 'micro':
                    metrics['auc'] = roc_auc_score(
                        np.eye(y_score.shape[1])[y_true], y_score,
                        average=average, multi_class='ovr'
                    ) * 100
                else:
                    metrics['auc'] = roc_auc_score(
                        np.eye(y_score.shape[1])[y_true], y_score,
                        average=average, multi_class='ovr'
                    ) * 100
            else:  # 二分类情况
                metrics['auc'] = roc_auc_score(y_true, y_score[:, 1]) * 100

                # PR曲线下面积
                precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
                metrics['pr_auc'] = auc(recall, precision) * 100
        except (ValueError, IndexError) as e:
            # 某些情况下可能无法计算AUC
            metrics['auc'] = np.nan

    return metrics


def mean_confidence_interval(data, confidence=0.95):
    """
    计算均值的置信区间

    参数:
    - data: 数据样本
    - confidence: 置信水平 (默认95%)

    返回:
    - mean: 均值
    - ci: 置信区间半径
    """
    import scipy.stats

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def meta_learning_metrics(task_accuracies):
    """
    计算元学习特定的指标

    参数:
    - task_accuracies: 多个任务的准确率列表

    返回:
    - 元学习指标字典
    """
    # 计算平均准确率和95%置信区间
    mean_acc, ci = mean_confidence_interval(task_accuracies)

    # 计算标准差、最大值和最小值
    std_acc = np.std(task_accuracies)
    min_acc = np.min(task_accuracies)
    max_acc = np.max(task_accuracies)

    # 计算任务间变异系数 (CoV)
    cov = std_acc / mean_acc if mean_acc > 0 else float('inf')

    # 统计超过一定阈值(例如90%)的任务比例
    threshold = 90.0  # 90%准确率阈值
    success_rate = np.mean(np.array(task_accuracies) > threshold) * 100

    return {
        'mean_accuracy': mean_acc,
        'confidence_interval': ci,
        'std_accuracy': std_acc,
        'min_accuracy': min_acc,
        'max_accuracy': max_acc,
        'coef_variation': cov,
        'success_rate': success_rate,  # 高准确率任务比例
        'num_tasks': len(task_accuracies)
    }


def few_shot_metrics(results, shot_values):
    """
    计算少样本学习实验的指标

    参数:
    - results: 不同shot设置下的结果字典
    - shot_values: 样本数列表(k-shot值)

    返回:
    - 少样本学习指标字典
    """
    metrics = {}

    # 计算不同shot下的性能对比
    for k in shot_values:
        if k in results:
            metrics[f'{k}-shot'] = results[k]

    # 计算样本效率 (sample efficiency)
    if len(shot_values) >= 2:
        # 假设最小和最大shot值对应的结果都存在
        min_shot = min(shot_values)
        max_shot = max(shot_values)

        if min_shot in results and max_shot in results:
            # 计算从min_shot到max_shot的性能提升
            min_acc = results[min_shot]['accuracy']
            max_acc = results[max_shot]['accuracy']

            # 样本效率 = 平均每增加一个样本带来的性能提升百分比
            shot_diff = max_shot - min_shot
            if shot_diff > 0:
                efficiency = (max_acc - min_acc) / shot_diff
                metrics['sample_efficiency'] = efficiency

            # 小样本敏感度 = 最高性能与最低性能的比例
            if min_acc > 0:
                metrics['shot_sensitivity'] = max_acc / min_acc

    return metrics


def ablation_contribution(baseline_acc, full_acc, component_accs):
    """
    计算不同组件的贡献度

    参数:
    - baseline_acc: 基线模型准确率
    - full_acc: 完整模型准确率
    - component_accs: 各个组件的准确率列表

    返回:
    - 各组件贡献度字典
    """
    # 总提升
    total_improvement = full_acc - baseline_acc

    if total_improvement <= 0:
        return {'contribution': {}, 'total_improvement': 0}

    # 计算相对贡献
    contributions = {}

    prev_acc = baseline_acc
    for i, (name, acc) in enumerate(component_accs.items()):
        component_gain = acc - prev_acc
        if total_improvement > 0:
            relative_contribution = (component_gain / total_improvement) * 100
        else:
            relative_contribution = 0

        contributions[name] = {
            'absolute_gain': component_gain,
            'relative_contribution': relative_contribution
        }

        prev_acc = acc

    return {
        'contribution': contributions,
        'total_improvement': total_improvement
    }