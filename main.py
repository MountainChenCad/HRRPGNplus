import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import copy
import pandas as pd
from tqdm import tqdm
from config import Config
from dataset import (
    prepare_datasets, TaskGenerator, HRRPTransform,
    visualize_dataset_statistics, visualize_data_augmentation,
    prepare_snr_test_data
)
from models import (
    MDGN, CNNModel, LSTMModel, GCNModel, GATModel,
    ProtoNetModel, MatchingNetModel, PCASVM, TemplateMatcher,
    StaticGraphModel, DynamicGraphModel, HybridGraphModel
)
from train import (
    MAMLPlusPlusTrainer as MAMLTrainer, test_model, run_shot_experiment,  # Update this line
    ablation_study_lambda, ablation_study_dynamic_graph, ablation_study_gnn_architecture,
    noise_robustness_experiment, compare_with_baselines, ablation_study_meta_learning,
    visualize_model_interpretability, computational_complexity_analysis, compare_models_across_shots
)
from utils import (
    plot_learning_curve, plot_shot_curve, plot_confusion_matrix,
    visualize_features, visualize_dynamic_graph, visualize_attention,
    compute_metrics, log_metrics, create_experiment_log, find_latest_experiment,
    analyze_model_complexity, visualize_model_parameters, create_experiment_report,
    create_model_summary, prepare_static_adjacency
)

# Define CVPR-quality color palette
COLORS = ['#0783D5', '#E52119', '#FD751F', '#0E2D88', '#78196D',
          '#C2C121', '#FC837E', '#00A6BC', '#025057', '#7E5505', '#77196C']

# Set matplotlib parameters for CVPR-quality plots
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


def parse_args():
    parser = argparse.ArgumentParser(description='HRRPGraphNet++ Experimental Framework')

    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'ablation', 'baseline', 'robustness', 'visualization', 'complexity'],
                        help='Operation mode: train, test, ablation, baseline, robustness, visualization, complexity')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='simulated', choices=['simulated', 'measured'],
                        help='Dataset type: simulated or measured')
    parser.add_argument('--cv', type=int, default=0,
                        help='Cross-validation scheme index')

    # Training parameters
    parser.add_argument('--shot', type=int, default=None,
                        help='K-shot setting for few-shot learning')
    parser.add_argument('--way', type=int, default=None,
                        help='N-way setting for few-shot learning')
    parser.add_argument('--query', type=int, default=None,
                        help='Query samples per class')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Task batch size')
    parser.add_argument('--inner_steps', type=int, default=None,
                        help='MAML inner loop steps')
    parser.add_argument('--inner_lr', type=float, default=None,
                        help='MAML inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=None,
                        help='MAML outer loop learning rate')

    # Model parameters
    parser.add_argument('--lambda_mix', type=float, default=None,
                        help='Static-dynamic graph mixing coefficient')
    parser.add_argument('--heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=None,
                        help='Number of graph convolutional layers')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')

    # Experiment configuration
    parser.add_argument('--ablation_type', type=str, default='all',
                        choices=['all', 'lambda', 'dynamic_graph', 'gnn_architecture', 'meta_learning'],
                        help='Type of ablation study to run')
    parser.add_argument('--baseline_models', type=str, nargs='+',
                        choices=['CNN', 'LSTM', 'GCN', 'GAT', 'ProtoNet', 'MatchingNet', 'PCA+SVM', 'Template'],
                        help='Baseline models to compare with')
    parser.add_argument('--test_shots', type=int, nargs='+', default=None,
                        help='K-shot values for shot experiment')
    parser.add_argument('--test_snr', type=float, nargs='+', default=None,
                        help='SNR values for noise robustness experiment')

    # Utility parameters
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (timestamp) for loading a previous model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Direct path to model file to load')
    parser.add_argument('--vis', action='store_true',
                        help='Enable detailed visualization')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--report', action='store_true',
                        help='Generate comprehensive experiment report')

    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration based on command-line arguments"""
    # Dataset configuration
    if args.dataset:
        Config.data_root = f'datasets/{args.dataset}'
        Config.train_dir = os.path.join(Config.data_root, 'train')
        Config.test_dir = os.path.join(Config.data_root, 'test')

    # Few-shot learning configuration
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.way is not None:
        Config.n_way = args.way
    if args.query is not None:
        Config.q_query = args.query

    # Training configuration
    if args.epochs is not None:
        Config.max_epochs = args.epochs
    if args.batch is not None:
        Config.task_batch_size = args.batch
    if args.inner_steps is not None:
        Config.inner_steps = args.inner_steps
    if args.inner_lr is not None:
        Config.inner_lr = args.inner_lr
    if args.outer_lr is not None:
        Config.outer_lr = args.outer_lr

    # Model configuration
    if args.lambda_mix is not None:
        Config.lambda_mix = args.lambda_mix
    if args.heads is not None:
        Config.attention_heads = args.heads
    if args.layers is not None:
        Config.graph_conv_layers = args.layers
    if args.dropout is not None:
        Config.dropout = args.dropout

    # Seed configuration
    if args.seed is not None:
        Config.seed = args.seed

    # Device configuration
    if args.device is not None:
        Config.device = torch.device(args.device)

    # Test experiment configuration
    if args.test_shots is not None:
        Config.data_sparsity['shot_levels'] = args.test_shots
    if args.test_snr is not None:
        Config.noise_robustness['snr_levels'] = args.test_snr

    print(f"Configuration updated from command-line arguments")


def train(args, vis=False):
    """Train HRRPGraphNet++ model"""
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient samples
    if not train_dataset.samples or not test_dataset.samples:
        print("Error: At least one dataset has no valid samples. Check data paths and file formats.")
        return None, 0

    # Set K-shot and q_query
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.query is not None:
        Config.q_query = args.query

    # Calculate minimum samples per class
    train_min_samples = float('inf')
    for class_idx in train_dataset.class_samples:
        samples_count = len(train_dataset.class_samples[class_idx])
        if samples_count < train_min_samples:
            train_min_samples = samples_count

    if train_min_samples == float('inf'):
        print("Error: Training dataset has no available samples.")
        return None, 0

    print(f"\nMinimum samples per class in training dataset: {train_min_samples}")

    # Automatically adjust k_shot and q_query if needed
    total_needed = Config.k_shot + Config.q_query
    if total_needed > train_min_samples:
        old_k_shot = Config.k_shot
        old_q_query = Config.q_query

        # Redistribute samples
        if train_min_samples <= 1:
            Config.k_shot = 1
            Config.q_query = 0
        else:
            # Try to keep at least one query sample
            Config.k_shot = min(old_k_shot, train_min_samples - 1)
            Config.q_query = train_min_samples - Config.k_shot

        print(
            f"Warning: Insufficient samples, adjusting parameters k_shot: {old_k_shot} -> {Config.k_shot}, q_query: {old_q_query} -> {Config.q_query}")

    # Ensure q_query is at least 1
    if Config.q_query < 1:
        print("Error: Not enough samples for query set. Each class needs at least k_shot+1 samples.")
        if train_min_samples >= 2:
            Config.k_shot = train_min_samples - 1
            Config.q_query = 1
            print(f"Automatically adjusting to: k_shot={Config.k_shot}, q_query={Config.q_query}")
        else:
            return None, 0

    print(f"\nFinal training parameters: {Config.k_shot}-shot, {Config.q_query}-query")

    # Create task generators
    train_task_generator = TaskGenerator(train_dataset, n_way=Config.train_n_way, k_shot=Config.k_shot,
                                         q_query=Config.q_query)
    val_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot,
                                       q_query=Config.q_query)

    # Create model and trainer
    model = MDGN(num_classes=Config.train_n_way)
    trainer = MAMLTrainer(model, Config.device)

    # Create experiment log
    create_experiment_log(Config)

    # Visualize dataset statistics if enabled
    if vis:
        print("\nGenerating dataset visualizations...")
        stats_save_path = os.path.join(Config.log_dir, 'dataset_statistics.png')
        visualize_dataset_statistics(train_dataset, save_path=stats_save_path)

        aug_save_path = os.path.join(Config.log_dir, 'data_augmentation.png')
        visualize_data_augmentation(train_dataset, save_path=aug_save_path)

    # Train model
    print(f"\nStarting training: {Config.k_shot}-shot, {Config.train_n_way}-way...")
    start_time = time.time()
    train_results = trainer.train(train_task_generator, val_task_generator, num_epochs=Config.max_epochs)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Plot learning curve
    if vis:
        plot_learning_curve(
            train_results['train_accs'],
            train_results['val_accs'],
            train_losses=train_results['train_losses'],
            val_losses=train_results['val_losses'],
            title=f"{Config.k_shot}-shot {Config.train_n_way}-way Learning Curve",
            save_path=os.path.join(Config.log_dir, 'learning_curve.png')
        )

    # Test best model
    print("\nTesting best model...")
    test_accuracy, test_ci, all_accuracies, test_f1 = test_model(model, val_task_generator, Config.device)
    print(f"Final test accuracy: {test_accuracy:.2f}% ± {test_ci:.2f}%, F1: {test_f1:.2f}%")

    # Save test results
    results_path = os.path.join(Config.log_dir, 'final_results.json')
    results = {
        'accuracy': test_accuracy,
        'confidence_interval': test_ci,
        'f1_score': test_f1,
        'training_time': training_time,
        'best_epoch': train_results['best_epoch'],
        'best_val_accuracy': train_results['best_val_accuracy'],
        'training_config': {
            'k_shot': Config.k_shot,
            'n_way': Config.train_n_way,
            'q_query': Config.q_query,
            'inner_steps': Config.inner_steps,
            'inner_lr': Config.inner_lr,
            'outer_lr': Config.outer_lr,
            'lambda_mix': Config.lambda_mix
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Analyze model if visualization is enabled
    if vis:
        print("\nAnalyzing model...")
        model_params_path = os.path.join(Config.log_dir, 'model_parameters.png')
        visualize_model_parameters(model, save_path=model_params_path)

        # Generate model summary
        model_summary_path = os.path.join(Config.log_dir, 'model_summary.txt')
        create_model_summary(model, save_path=model_summary_path)

    return model, test_accuracy


def test(args, model=None, vis=False):
    """Test HRRPGraphNet++ model with comprehensive evaluation"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return None
    # Try to find latest experiment if no model passed directly
    elif model is None:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    # Prepare datasets
    _, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient test samples
    if not test_dataset.samples:
        print("Error: Test dataset has no valid samples. Check data paths and file formats.")
        return None

    # Set K-shot and q_query
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.query is not None:
        Config.q_query = args.query

    # Calculate minimum samples per class
    test_min_samples = float('inf')
    for class_idx in test_dataset.class_samples:
        samples_count = len(test_dataset.class_samples[class_idx])
        if samples_count < test_min_samples:
            test_min_samples = samples_count

    if test_min_samples == float('inf'):
        print("Error: Test dataset has no available samples.")
        return None

    print(f"\nMinimum samples per class in test dataset: {test_min_samples}")

    # Automatically adjust k_shot and q_query if needed
    total_needed = Config.k_shot + Config.q_query
    if total_needed > test_min_samples:
        old_k_shot = Config.k_shot
        old_q_query = Config.q_query

        # Redistribute samples
        if test_min_samples <= 1:
            Config.k_shot = 1
            Config.q_query = 0
        else:
            # Try to keep at least one query sample
            Config.k_shot = min(old_k_shot, test_min_samples - 1)
            Config.q_query = test_min_samples - Config.k_shot

        print(
            f"Warning: Insufficient samples, adjusting parameters k_shot: {old_k_shot} -> {Config.k_shot}, q_query: {old_q_query} -> {Config.q_query}")

    # Ensure q_query is at least 1
    if Config.q_query < 1:
        print("Error: Not enough samples for query set. Each class needs at least k_shot+1 samples.")
        if test_min_samples >= 2:
            Config.k_shot = test_min_samples - 1
            Config.q_query = 1
            print(f"Automatically adjusting to: k_shot={Config.k_shot}, q_query={Config.q_query}")
        else:
            return None

    print(f"\nFinal test parameters: {Config.k_shot}-shot, {Config.q_query}-query")

    # Set up task generator
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot,
                                        q_query=Config.q_query)

    # Load or use provided model
    if model is None:
        model = MDGN(num_classes=Config.test_n_way)
        model = model.to(Config.device)

        # Prioritize directly specified model path
        if args.model_path and os.path.exists(args.model_path):
            model_path = args.model_path
        else:
            model_path = os.path.join(Config.save_dir, 'best_model.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model: {model_path}")
        else:
            print(f"Model not found: {model_path}. Please train first or provide correct path.")
            return None

    # Create results directory
    results_dir = os.path.join(Config.log_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)

    # Perform standard test
    print(f"\nTesting configuration: {Config.k_shot}-shot, {Config.test_n_way}-way...")
    test_accuracy, test_ci, all_accuracies, test_f1 = test_model(model, test_task_generator, Config.device)
    print(f"Test accuracy: {test_accuracy:.2f}% ± {test_ci:.2f}%, F1: {test_f1:.2f}%")

    # Save test results
    test_results = {
        'accuracy': test_accuracy,
        'confidence_interval': test_ci,
        'f1_score': test_f1,
        'test_config': {
            'k_shot': Config.k_shot,
            'n_way': Config.test_n_way,
            'q_query': Config.q_query
        }
    }

    with open(os.path.join(results_dir, 'standard_test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

        # Perform shot experiment
        if vis:
            print("\nRunning shot experiment...")
            # Determine feasible shot sizes
            max_shot = max(1, test_min_samples - 1)  # At least 1 sample for query

            # Adjust shot range based on available samples
            all_shots = [1, 5, 10, 20]
            feasible_shots = [s for s in all_shots if s < max_shot]
            if not feasible_shots:
                feasible_shots = [1]  # Use at least 1-shot

            print(f"Feasible shot values: {feasible_shots}")

            # Initialize baseline models for comparison if requested
            if args.baseline_models:
                print("\nRunning comparison with baseline models...")

                # Create comparison directory
                comparison_dir = os.path.join(results_dir, 'model_comparison')
                os.makedirs(comparison_dir, exist_ok=True)

                # Run model comparison across different shot values
                comparison_results = compare_models_across_shots(
                    test_dataset, Config.device,
                    baseline_models=args.baseline_models,
                    shot_sizes=feasible_shots
                )

                # Save comparison results
                with open(os.path.join(comparison_dir, 'model_comparison_results.json'), 'w') as f:
                    json.dump(comparison_results, f, indent=4)

                print(f"Model comparison results saved to {comparison_dir}")
            else:
                # Run standard shot experiment for just the main model
                shot_sizes, shot_results, shot_ci, shot_f1 = run_shot_experiment(  # Use the new function
                    model, test_task_generator, Config.device,
                    shot_sizes=feasible_shots
                )

                # Plot shot curve with error bars
                plot_shot_curve(
                    shot_sizes, shot_results, ci=shot_ci, f1_scores=shot_f1,
                    title="Performance vs Number of Shots",
                    save_path=os.path.join(results_dir, 'shot_curve.png')
                )

                # Save shot experiment results
                shot_exp_results = {
                    'shot_sizes': shot_sizes,
                    'accuracies': shot_results,
                    'confidence_intervals': shot_ci,
                    'f1_scores': shot_f1
                }

                with open(os.path.join(results_dir, 'shot_experiment_results.json'), 'w') as f:
                    json.dump(shot_exp_results, f, indent=4)

            # Create experiment visualizations
            print("\nGenerating model interpretability visualizations...")
            visualize_model_interpretability(model, test_task_generator, Config.device)

    # Generate summary report if requested
    if args.report:
        print("\nGenerating experiment report...")
        report_data = {
            'metrics': {
                'accuracy': test_accuracy,
                'f1': test_f1,
                'confidence_interval': test_ci
            },
            'learning_curves': 'learning_curve.png',
            'confusion_matrix': 'test_results/confusion_matrix.png',
            'visualizations': {
                'Shot Experiment': 'test_results/shot_curve.png',
                'Attention Weights': 'visualizations/attention/attention_sample_1.png',
                'Dynamic Graph': 'visualizations/dynamic_graph/dynamic_graph_structure.png',
                'Feature Space': 'visualizations/features/feature_space.png'
            }
        }

        create_experiment_report(
            experiment_id=args.exp_id or find_latest_experiment(),
            results=report_data
        )

    return test_accuracy


def run_ablation_studies(args):
    """Run comprehensive ablation experiments"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return
    # Try to find latest experiment
    else:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    # Prepare datasets
    _, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient test samples
    if not test_dataset.samples:
        print("Error: Test dataset has no valid samples. Check data paths and file formats.")
        return

    # Set K-shot and q_query
    if args.shot is not None:
        Config.k_shot = args.shot
    if args.query is not None:
        Config.q_query = args.query

    # Calculate minimum samples per class
    test_min_samples = float('inf')
    for class_idx in test_dataset.class_samples:
        samples_count = len(test_dataset.class_samples[class_idx])
        if samples_count < test_min_samples:
            test_min_samples = samples_count

    if test_min_samples == float('inf'):
        print("Error: Test dataset has no available samples.")
        return

    print(f"\nMinimum samples per class in test dataset: {test_min_samples}")

    # Automatically adjust k_shot and q_query if needed
    total_needed = Config.k_shot + Config.q_query
    if total_needed > test_min_samples:
        old_k_shot = Config.k_shot
        old_q_query = Config.q_query

        # Redistribute samples
        if test_min_samples <= 1:
            print("Error: Test dataset has insufficient samples for ablation studies")
            return
        else:
            # Try to keep at least one query sample
            Config.k_shot = min(old_k_shot, test_min_samples - 1)
            Config.q_query = test_min_samples - Config.k_shot

        print(
            f"Warning: Insufficient samples, adjusting parameters k_shot: {old_k_shot} -> {Config.k_shot}, q_query: {old_q_query} -> {Config.q_query}")

    print(f"\nFinal ablation study parameters: {Config.k_shot}-shot, {Config.q_query}-query")

    # Set up task generator
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot,
                                        q_query=Config.q_query)

    # Load model
    model = MDGN(num_classes=Config.test_n_way)
    model = model.to(Config.device)
    model_path = os.path.join(Config.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found: {model_path}. Please train first.")
        return

    # Create ablation results directory
    ablation_dir = os.path.join(Config.log_dir, 'ablation_studies')
    os.makedirs(ablation_dir, exist_ok=True)

    # Determine which ablation studies to run
    ablation_type = args.ablation_type

    # Run lambda mixing coefficient ablation study
    if ablation_type in ['all', 'lambda']:
        print("\nRunning lambda mixing coefficient ablation study...")
        lambda_values, lambda_results, lambda_ci, lambda_f1 = ablation_study_lambda(
            model, test_task_generator, Config.device
        )

        # Plot results with CVPR-quality styling
        plt.figure(figsize=(10, 6), facecolor='white')
        ax = plt.subplot(111)

        # Plot accuracy with error bars
        ax.errorbar(lambda_values, lambda_results, yerr=lambda_ci, fmt='o-',
                    capsize=5, linewidth=2.5, markersize=8, color=COLORS[0],
                    ecolor=COLORS[0], elinewidth=1.5, label='Accuracy')

        # Plot F1 score
        ax.plot(lambda_values, lambda_f1, 's--', linewidth=2.5, markersize=8,
                color=COLORS[1], label='F1 Score')

        # Styling
        ax.set_title("Effect of Lambda Mixing Coefficient", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Lambda (Static Graph Weight)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Performance (%)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=1.5, length=5)

        plt.tight_layout()
        plt.savefig(os.path.join(ablation_dir, 'lambda_ablation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save results
        lambda_results_data = {
            'lambda_values': lambda_values,
            'accuracies': lambda_results,
            'confidence_intervals': lambda_ci,
            'f1_scores': lambda_f1
        }

        with open(os.path.join(ablation_dir, 'lambda_ablation_results.json'), 'w') as f:
            json.dump(lambda_results_data, f, indent=4)

    # Run dynamic graph ablation study
    if ablation_type in ['all', 'dynamic_graph']:
        print("\nRunning dynamic graph ablation study...")
        graph_labels, graph_results, graph_ci, graph_f1 = ablation_study_dynamic_graph(
            model, test_task_generator, Config.device
        )

        # Plot results with CVPR-quality styling
        plt.figure(figsize=(10, 6), facecolor='white')
        ax = plt.subplot(111)
        x_pos = np.arange(len(graph_labels))

        # Set width for bars
        width = 0.35

        # Plot accuracy bars
        accuracy_bars = ax.bar(x_pos - width / 2, graph_results, width=width,
                               yerr=graph_ci, capsize=5, color=COLORS[0],
                               alpha=0.8, label='Accuracy', edgecolor='black', linewidth=1.5)

        # Plot F1 bars
        f1_bars = ax.bar(x_pos + width / 2, graph_f1, width=width,
                         color=COLORS[1], alpha=0.8, label='F1 Score',
                         edgecolor='black', linewidth=1.5)

        # Add data labels on top of bars
        for i, bar in enumerate(accuracy_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        for i, bar in enumerate(f1_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(graph_labels, fontsize=10, fontweight='bold')
        ax.set_title('Performance Comparison: Graph Structure Types',
                     fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=1.5, length=5)

        plt.tight_layout()
        plt.savefig(os.path.join(ablation_dir, 'dynamic_graph_ablation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save results
        graph_results_data = {
            'graph_labels': graph_labels,
            'accuracies': graph_results,
            'confidence_intervals': graph_ci,
            'f1_scores': graph_f1
        }

        with open(os.path.join(ablation_dir, 'dynamic_graph_ablation_results.json'), 'w') as f:
            json.dump(graph_results_data, f, indent=4)

    # Run meta-learning ablation study
    if ablation_type in ['all', 'meta_learning']:
        print("\nRunning meta-learning ablation study...")
        meta_learning_results = ablation_study_meta_learning(
            model, test_task_generator, Config.device
        )

        # Save results
        with open(os.path.join(ablation_dir, 'meta_learning_ablation_results.json'), 'w') as f:
            # 元学习结果是列表，需要正确处理
            serializable_results = []
            for item in meta_learning_results:
                serializable_item = {}
                for key, value in item.items():
                    if isinstance(value, np.ndarray):
                        serializable_item[key] = value.tolist()
                    else:
                        serializable_item[key] = value
                serializable_results.append(serializable_item)

            json.dump(serializable_results, f, indent=4)

    # Run GNN architecture ablation study
    if ablation_type in ['all', 'gnn_architecture']:
        print("\nRunning GNN architecture ablation study...")
        gnn_results = ablation_study_gnn_architecture(
            model, test_task_generator, Config.device
        )

        # Save results
        with open(os.path.join(ablation_dir, 'gnn_architecture_ablation_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in gnn_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value

            json.dump(serializable_results, f, indent=4)

    # Run noise robustness experiment
    print("\nRunning noise robustness experiment...")
    noise_levels, noise_results, noise_ci, noise_f1 = noise_robustness_experiment(
        model, test_task_generator, Config.device
    )

    # Plot results with CVPR-quality styling
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.subplot(111)

    # Plot accuracy with error bars
    ax.errorbar(noise_levels, noise_results, yerr=noise_ci, fmt='o-',
                capsize=5, linewidth=2.5, markersize=8, color=COLORS[0],
                ecolor=COLORS[0], elinewidth=1.5, label='Accuracy')

    # Plot F1 score
    ax.plot(noise_levels, noise_f1, 's--', linewidth=2.5, markersize=8,
            color=COLORS[1], label='F1 Score')

    # Styling
    ax.set_title("Noise Robustness Experiment", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("SNR (dB)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Performance (%)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Add data points labels
    for i, (x, y) in enumerate(zip(noise_levels, noise_results)):
        ax.annotate(f'{y:.1f}%', (x, y), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    noise_results_data = {
        'noise_levels': noise_levels,
        'accuracies': noise_results,
        'confidence_intervals': noise_ci,
        'f1_scores': noise_f1
    }

    with open(os.path.join(ablation_dir, 'noise_robustness_results.json'), 'w') as f:
        json.dump(noise_results_data, f, indent=4)


def compare_baseline_models(args):
    """Compare HRRPGraphNet++ with baseline methods"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return
    # Try to find latest experiment
    else:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    # Prepare datasets
    _, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient test samples
    if not test_dataset.samples:
        print("Error: Test dataset has no valid samples. Check data paths and file formats.")
        return

    # Set K-shot
    if args.shot is not None:
        Config.k_shot = args.shot

    # Determine baseline models to compare
    baseline_models = args.baseline_models
    if baseline_models is None:
        # Use all available methods from config
        traditional_models = Config.traditional_baselines['methods'] if Config.traditional_baselines['enabled'] else []
        dl_models = Config.dl_baselines['methods'] if Config.dl_baselines['enabled'] else []
        fsl_models = Config.fsl_baselines['methods'] if Config.fsl_baselines['enabled'] else []

        baseline_models = traditional_models + dl_models + fsl_models

    # Create baseline results directory
    baseline_dir = os.path.join(Config.log_dir, 'baseline_comparison')
    os.makedirs(baseline_dir, exist_ok=True)

    # Run baseline comparison
    print(f"\nComparing with baseline models using {Config.k_shot}-shot setting...")
    comparison_results = compare_with_baselines(
        test_dataset, Config.device, shot=Config.k_shot,
        baseline_models=baseline_models
    )

    # Save comparison results
    comparison_results.to_csv(os.path.join(baseline_dir, f'baseline_comparison_{Config.k_shot}shot.csv'), index=False)

    # Create bar chart visualization
    if comparison_results is not None and not comparison_results.empty:
        # Extract accuracy values (removing % and ± parts)
        accuracy_values = []
        model_names = []
        ci_values = []

        for _, row in comparison_results.iterrows():
            model_names.append(row['model'])
            acc_text = row['accuracy']
            acc_parts = acc_text.split('±')
            acc_value = float(acc_parts[0].strip().replace('%', ''))
            ci_value = float(acc_parts[1].strip().replace('%', '')) if len(acc_parts) > 1 else 0
            accuracy_values.append(acc_value)
            ci_values.append(ci_value)

        # Create bar chart
        plt.figure(figsize=(12, 6), facecolor='white')
        x_pos = np.arange(len(model_names))

        ax = plt.subplot(111)
        bars = ax.bar(x_pos, accuracy_values, yerr=ci_values, capsize=5,
                      color=[COLORS[i % len(COLORS)] for i in range(len(model_names))],
                      width=0.6, edgecolor='black', linewidth=1.5)

        # Add accuracy values on top of bars
        for i, (bar, value) in enumerate(zip(bars, accuracy_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + ci_values[i] + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_title(f'Model Comparison ({Config.k_shot}-shot)', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=1.5, length=5)

        plt.tight_layout()
        plt.savefig(os.path.join(baseline_dir, f'model_comparison_{Config.k_shot}shot.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    print(f"\nBaseline comparison results saved to {baseline_dir}")

    # If visualization is enabled, also run shot experiment for multiple models
    if args.vis and baseline_models:
        # Run shot comparison across different k values
        shot_dir = os.path.join(baseline_dir, 'shot_comparison')
        os.makedirs(shot_dir, exist_ok=True)

        # Determine feasible shot sizes
        test_min_samples = float('inf')
        for class_idx in test_dataset.class_samples:
            samples_count = len(test_dataset.class_samples[class_idx])
            if samples_count < test_min_samples:
                test_min_samples = samples_count

        max_shot = max(1, test_min_samples - 1)  # At least 1 sample for query
        all_shots = [1, 5, 10, 20]
        feasible_shots = [s for s in all_shots if s < max_shot]
        if not feasible_shots:
            feasible_shots = [1]  # Use at least 1-shot

        print(f"\nRunning shot comparison with sizes: {feasible_shots}")

        # Prepare results with shot experiment - use the renamed function
        shot_results = compare_models_across_shots(
            test_dataset, Config.device,
            baseline_models=baseline_models,
            shot_sizes=feasible_shots
        )

        # Save shot comparison results
        with open(os.path.join(shot_dir, 'shot_comparison_results.json'), 'w') as f:
            json.dump(shot_results, f, indent=4)

        print(f"Shot comparison results saved to {shot_dir}")


def run_robustness_analysis(args):
    """Run comprehensive robustness analysis"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return
    # Try to find latest experiment
    else:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    # Prepare datasets
    _, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient test samples
    if not test_dataset.samples:
        print("Error: Test dataset has no valid samples. Check data paths and file formats.")
        return

    # Set K-shot if specified
    if args.shot is not None:
        Config.k_shot = args.shot

    # Create robustness results directory
    robustness_dir = os.path.join(Config.log_dir, 'robustness_analysis')
    os.makedirs(robustness_dir, exist_ok=True)

    # Load model
    model = MDGN(num_classes=Config.test_n_way)
    model = model.to(Config.device)
    model_path = os.path.join(Config.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found: {model_path}. Please train first.")
        return

    # Set SNR levels for testing
    snr_levels = Config.noise_robustness['snr_levels']
    if args.test_snr is not None:
        snr_levels = args.test_snr

    # Prepare task generator
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot)

    # Run noise robustness test
    print(f"\nRunning noise robustness analysis at SNR levels: {snr_levels}...")
    snr_test_tasks = prepare_snr_test_data(test_dataset, snr_levels=snr_levels)

    # Test model at each SNR level
    snr_results = []
    snr_ci = []
    snr_f1 = []

    for snr in snr_levels:
        print(f"\nTesting at SNR={snr}dB...")

        # Skip if no tasks for this SNR level
        if snr not in snr_test_tasks or not snr_test_tasks[snr]:
            print(f"No tasks available for SNR={snr}dB, skipping")
            snr_results.append(0)
            snr_ci.append(0)
            snr_f1.append(0)
            continue

        # Test on pre-generated noisy tasks
        tasks = snr_test_tasks[snr]
        all_accs = []
        all_f1s = []
        all_preds = []
        all_labels = []

        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(tqdm(tasks, desc=f"SNR={snr}dB")):
            # Move data to device
            support_x, support_y = support_x.to(Config.device), support_y.to(Config.device)
            query_x, query_y = query_x.to(Config.device), query_y.to(Config.device)

            # Fine-tune model on support set (MAML inner loop)
            temp_trainer = MAMLTrainer(copy.deepcopy(model), Config.device)
            updated_model, _, _, _ = temp_trainer.inner_loop(support_x, support_y)

            # Test on query set
            batch_size, channels, seq_len = query_x.shape
            static_adj = prepare_static_adjacency(batch_size, seq_len, Config.device)

            with torch.no_grad():
                logits, _ = updated_model(query_x, static_adj)
                preds = torch.argmax(logits, dim=1)

                # Calculate accuracy
                acc = (preds == query_y).float().mean().item()
                all_accs.append(acc)

                # Collect predictions and labels for F1 calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(query_y.cpu().numpy())

        # Calculate metrics
        avg_acc = np.mean(all_accs) * 100
        std_acc = np.std(all_accs) * 100
        ci = 1.96 * std_acc / np.sqrt(len(all_accs))

        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro') * 100

        print(f"SNR={snr}dB: Accuracy={avg_acc:.2f}% ± {ci:.2f}%, F1={f1:.2f}%")

        snr_results.append(avg_acc)
        snr_ci.append(ci)
        snr_f1.append(f1)

    # Plot SNR vs Accuracy curve with CVPR-quality styling
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.subplot(111)

    # Plot accuracy with error bars
    ax.errorbar(snr_levels, snr_results, yerr=snr_ci, fmt='o-',
                capsize=5, linewidth=2.5, markersize=8, color=COLORS[0],
                ecolor=COLORS[0], elinewidth=1.5, label='Accuracy')

    # Plot F1 score
    ax.plot(snr_levels, snr_f1, 's--', linewidth=2.5, markersize=8,
            color=COLORS[1], label='F1 Score')

    # Styling
    ax.set_title("Noise Robustness Analysis", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Performance (%)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=1.5, length=5)

    # Add data point labels
    for i, (x, y) in enumerate(zip(snr_levels, snr_results)):
        ax.annotate(f'{y:.1f}%', (x, y), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(robustness_dir, 'snr_robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    snr_results_data = {
        'snr_levels': snr_levels,
        'accuracies': snr_results,
        'confidence_intervals': snr_ci,
        'f1_scores': snr_f1
    }

    with open(os.path.join(robustness_dir, 'snr_robustness_results.json'), 'w') as f:
        json.dump(snr_results_data, f, indent=4)

    print(f"\nNoise robustness analysis completed. Results saved to {robustness_dir}")


def run_visualization_analysis(args):
    """Run comprehensive visualization and interpretability analysis"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return
    # Try to find latest experiment
    else:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    # Prepare datasets
    _, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient test samples
    if not test_dataset.samples:
        print("Error: Test dataset has no valid samples. Check data paths and file formats.")
        return

    # Set K-shot if specified
    if args.shot is not None:
        Config.k_shot = args.shot

    # Create visualization directory
    vis_dir = os.path.join(Config.log_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Load model
    model = MDGN(num_classes=Config.test_n_way)
    model = model.to(Config.device)
    model_path = os.path.join(Config.save_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found: {model_path}. Please train first.")
        return

    # Prepare task generator
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot)

    # Run model interpretability visualizations
    print("\nGenerating model interpretability visualizations...")
    visualize_model_interpretability(model, test_task_generator, Config.device)

    # Visualize dataset statistics
    print("\nVisualizing dataset statistics...")
    stats_save_path = os.path.join(vis_dir, 'dataset_statistics.png')
    visualize_dataset_statistics(test_dataset, save_path=stats_save_path)

    # Visualize model parameters
    print("\nVisualizing model parameters...")
    params_save_path = os.path.join(vis_dir, 'model_parameters.png')
    visualize_model_parameters(model, save_path=params_save_path)

    # Generate model summary
    print("\nGenerating model summary...")
    summary_save_path = os.path.join(vis_dir, 'model_summary.txt')
    create_model_summary(model, save_path=summary_save_path)

    print(f"\nVisualization analysis completed. Results saved to {vis_dir}")


def run_complexity_analysis(args):
    """Run computational complexity analysis"""
    # Load experiment config if ID provided
    if args.exp_id:
        if not Config.load_experiment(args.exp_id):
            print(f"Error: Could not load experiment {args.exp_id}")
            return
    # Try to find latest experiment
    else:
        latest_exp = find_latest_experiment()
        if latest_exp:
            Config.load_experiment(latest_exp)
            print(f"Automatically loaded latest experiment: {latest_exp}")

    # Prepare datasets
    _, test_dataset = prepare_datasets(scheme_idx=args.cv, dataset_type=args.dataset)

    # Check for sufficient test samples
    if not test_dataset.samples:
        print("Error: Test dataset has no valid samples. Check data paths and file formats.")
        return

    # Create complexity results directory
    complexity_dir = os.path.join(Config.log_dir, 'computational_complexity')
    os.makedirs(complexity_dir, exist_ok=True)

    # Prepare task generator
    test_task_generator = TaskGenerator(test_dataset, n_way=Config.test_n_way, k_shot=Config.k_shot)

    # Generate a sample task for analysis
    print("\nGenerating sample task for complexity analysis...")
    try:
        sample_task = test_task_generator.generate_task()
    except Exception as e:
        print(f"Error generating task: {e}. Please check dataset.")
        return

    # Run computational complexity analysis
    print("\nRunning computational complexity analysis...")
    complexity_results = computational_complexity_analysis(test_task_generator, Config.device, save_dir=complexity_dir)

    print("\nComplexity analysis results:")
    results_df = pd.DataFrame(complexity_results)
    print(results_df.to_string(index=False))
    print(f"\nComplexity analysis completed. Results saved to {complexity_dir}")


def main():
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        Config.seed = args.seed
    Config.set_seed()

    # Update configuration from command-line arguments
    update_config_from_args(args)

    # Only create new directories in training mode
    if args.mode == 'train':
        Config.create_directories()

    try:
        # Execute based on mode
        if args.mode == 'train':
            model, _ = train(args, vis=args.vis)
            if model is not None:  # Only test if training was successful
                test(args, model=model, vis=args.vis)
        elif args.mode == 'test':
            test(args, vis=args.vis)
        elif args.mode == 'ablation':
            run_ablation_studies(args)
        elif args.mode == 'baseline':
            compare_baseline_models(args)
        elif args.mode == 'robustness':
            run_robustness_analysis(args)
        elif args.mode == 'visualization':
            run_visualization_analysis(args)
        elif args.mode == 'complexity':
            run_complexity_analysis(args)
        else:
            print(f"Unknown mode: {args.mode}")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check dataset paths and file formats.")


if __name__ == "__main__":
    main()