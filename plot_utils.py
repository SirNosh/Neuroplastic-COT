import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_training_metrics(metrics_path: str) -> Dict:
    """Load training metrics from JSON file."""
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found: {metrics_path}")
        return None

    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_training_loss(metrics: Dict, title: str, save_path: Optional[str] = None):
    """
    Plot training loss curve.

    Args:
        metrics: Dictionary containing training metrics
        title: Plot title
        save_path: Path to save the plot (if None, displays the plot)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = metrics['steps']
    losses = metrics['losses']

    ax.plot(steps, losses, label='Training Loss', linewidth=2, alpha=0.7)

    # Add smoothed line
    if len(losses) > 10:
        window = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        ax.plot(smoothed_steps, smoothed, label='Smoothed Loss', linewidth=2, color='red')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_learning_rate(metrics: Dict, title: str, save_path: Optional[str] = None):
    """
    Plot learning rate schedule.

    Args:
        metrics: Dictionary containing training metrics
        title: Plot title
        save_path: Path to save the plot (if None, displays the plot)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = metrics['steps']
    lrs = metrics['learning_rates']

    ax.plot(steps, lrs, linewidth=2, color='green')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def compare_training_losses(metrics_dict: Dict[str, Dict], title: str = "Training Loss Comparison",
                           save_path: Optional[str] = None):
    """
    Compare training losses across multiple models.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = sns.color_palette("husl", len(metrics_dict))

    for (name, metrics), color in zip(metrics_dict.items(), colors):
        steps = metrics['steps']
        losses = metrics['losses']

        # Plot raw loss
        ax.plot(steps, losses, label=f'{name}', alpha=0.3, color=color, linewidth=1)

        # Plot smoothed loss
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            ax.plot(smoothed_steps, smoothed, label=f'{name} (smoothed)',
                   linewidth=2.5, color=color)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_eval_losses(metrics_dict: Dict[str, Dict], title: str = "Evaluation Loss Comparison",
                    save_path: Optional[str] = None):
    """
    Compare evaluation losses across multiple models.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("husl", len(metrics_dict))

    for (name, metrics), color in zip(metrics_dict.items(), colors):
        if 'eval_losses' in metrics and metrics['eval_losses']:
            eval_data = metrics['eval_losses']
            steps = [item['step'] for item in eval_data]
            eval_losses = [item['eval_loss'] for item in eval_data]

            ax.plot(steps, eval_losses, label=name, marker='o',
                   linewidth=2, markersize=6, color=color)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Evaluation Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved eval loss plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_ewc_loss(metrics: Dict, title: str = "EWC Loss Over Training",
                 save_path: Optional[str] = None):
    """
    Plot EWC loss over training.

    Args:
        metrics: Dictionary containing training metrics with EWC losses
        title: Plot title
        save_path: Path to save the plot
    """
    if 'ewc_losses' not in metrics or not metrics['ewc_losses']:
        logger.warning("No EWC losses found in metrics")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ewc_losses = metrics['ewc_losses']
    # EWC losses are recorded at same intervals as regular losses
    steps = metrics['steps'][:len(ewc_losses)]

    ax.plot(steps, ewc_losses, linewidth=2, color='purple', alpha=0.7)

    # Add smoothed line
    if len(ewc_losses) > 10:
        window = min(50, len(ewc_losses) // 10)
        smoothed = np.convolve(ewc_losses, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        ax.plot(smoothed_steps, smoothed, label='Smoothed EWC Loss',
               linewidth=2, color='darkviolet')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('EWC Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved EWC loss plot to {save_path}")
        plt.close()
    else:
        plt.show()


def create_dashboard(metrics_dict: Dict[str, Dict], output_dir: str):
    """
    Create a comprehensive dashboard with multiple plots.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training loss comparison
    compare_training_losses(
        metrics_dict,
        title="Training Loss Comparison Across Models",
        save_path=os.path.join(output_dir, "training_loss_comparison.png")
    )

    # 2. Evaluation loss comparison
    plot_eval_losses(
        metrics_dict,
        title="Evaluation Loss Comparison",
        save_path=os.path.join(output_dir, "eval_loss_comparison.png")
    )

    # 3. Individual training loss plots
    for name, metrics in metrics_dict.items():
        plot_training_loss(
            metrics,
            title=f"{name.capitalize()} Model - Training Loss",
            save_path=os.path.join(output_dir, f"{name}_training_loss.png")
        )

        plot_learning_rate(
            metrics,
            title=f"{name.capitalize()} Model - Learning Rate Schedule",
            save_path=os.path.join(output_dir, f"{name}_learning_rate.png")
        )

    # 4. EWC loss if available
    for name, metrics in metrics_dict.items():
        if 'ewc_losses' in metrics and metrics['ewc_losses']:
            plot_ewc_loss(
                metrics,
                title=f"{name.capitalize()} Model - EWC Loss",
                save_path=os.path.join(output_dir, f"{name}_ewc_loss.png")
            )

    logger.info(f"Dashboard created successfully in {output_dir}")


def plot_final_comparison_bar(metrics_dict: Dict[str, Dict], metric_name: str = "final_loss",
                              title: str = "Final Loss Comparison", save_path: Optional[str] = None):
    """
    Create a bar chart comparing final metric values.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        metric_name: Name of the metric to compare
        title: Plot title
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(metrics_dict.keys())
    values = []

    for name in names:
        metrics = metrics_dict[name]
        if metric_name == "final_loss" and 'losses' in metrics:
            # Take average of last 10 losses
            final_losses = metrics['losses'][-10:]
            values.append(np.mean(final_losses))
        elif metric_name in metrics:
            values.append(metrics[metric_name])
        else:
            values.append(0)

    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bar chart to {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--metrics_files", nargs="+", required=True,
                       help="Paths to metrics JSON files")
    parser.add_argument("--model_names", nargs="+", required=True,
                       help="Names for each model")
    parser.add_argument("--output_dir", type=str, default="./plots",
                       help="Directory to save plots")

    args = parser.parse_args()

    if len(args.metrics_files) != len(args.model_names):
        raise ValueError("Number of metrics files must match number of model names")

    # Load all metrics
    metrics_dict = {}
    for metrics_file, model_name in zip(args.metrics_files, args.model_names):
        metrics = load_training_metrics(metrics_file)
        if metrics:
            metrics_dict[model_name] = metrics

    if metrics_dict:
        create_dashboard(metrics_dict, args.output_dir)
        plot_final_comparison_bar(
            metrics_dict,
            title="Final Loss Comparison",
            save_path=os.path.join(args.output_dir, "final_loss_comparison.png")
        )
    else:
        logger.error("No valid metrics found")
