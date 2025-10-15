"""
Visualization tools for continual learning experiments.

Creates forgetting curves, performance matrices, and mechanism contribution analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_continual_results(results_path: str) -> Dict:
    """Load continual learning results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_forgetting_curve(results_dict: Dict[str, Dict], output_path: str):
    """
    Plot forgetting curves showing performance retention across phases.

    Args:
        results_dict: Dictionary mapping model names to their results
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    phases = ["arithmetic", "algebra", "geometry"]
    colors = sns.color_palette("husl", len(results_dict))

    for (model_name, results), color in zip(results_dict.items(), colors):
        perf_matrix = results["performance_matrix"]

        # For each task, track performance after each subsequent phase
        for task_idx, task_name in enumerate(phases):
            performances = []
            x_positions = []

            for phase_idx in range(task_idx, len(phases)):
                phase_name = phases[phase_idx]
                key = f"after_{phase_name}_test_{task_name}"

                if key in perf_matrix:
                    # Convert loss to "accuracy-like" metric (lower loss = better)
                    # Using 1/loss as proxy for performance
                    loss = perf_matrix[key]
                    performance = 100 / (1 + loss)  # Normalized performance metric
                    performances.append(performance)
                    x_positions.append(phase_idx)

            if performances:
                # Plot this task's retention curve
                label = f"{model_name} - {task_name}" if task_idx == 0 else None
                ax.plot(x_positions, performances,
                       marker='o', linewidth=2.5, markersize=8,
                       color=color, alpha=0.7, label=label)

    ax.set_xlabel('Training Phase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (higher = better)', fontsize=14, fontweight='bold')
    ax.set_title('Task Performance Across Continual Learning Phases\n(Forgetting Curves)',
                fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([p.capitalize() for p in phases])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Forgetting curve saved to {output_path}")


def plot_forgetting_heatmap(results_dict: Dict[str, Dict], output_path: str):
    """
    Create heatmap showing forgetting percentages.

    Args:
        results_dict: Dictionary mapping model names to their results
        output_path: Path to save the plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, results) in zip(axes, results_dict.items()):
        forgetting_scores = results.get("forgetting_scores", {})

        # Create matrix
        tasks = list(forgetting_scores.keys())
        if not tasks:
            continue

        forgetting_matrix = []
        for task in tasks:
            forgetting_pct = forgetting_scores[task]["forgetting_percentage"]
            forgetting_matrix.append([forgetting_pct])

        # Plot heatmap
        sns.heatmap(forgetting_matrix, annot=True, fmt=".1f",
                   cmap="RdYlGn_r", center=0,
                   yticklabels=[t.capitalize() for t in tasks],
                   xticklabels=["Forgetting %"],
                   ax=ax, cbar_kws={'label': 'Forgetting %'})

        ax.set_title(f"{model_name.capitalize()} Model\nForgetting Scores",
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Forgetting heatmap saved to {output_path}")


def plot_performance_matrix(results_dict: Dict[str, Dict], output_path: str):
    """
    Create performance matrix showing accuracy on each task after each phase.

    Args:
        results_dict: Dictionary mapping model names to their results
        output_path: Path to save the plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))

    if n_models == 1:
        axes = [axes]

    phases = ["arithmetic", "algebra", "geometry"]

    for ax, (model_name, results) in zip(axes, results_dict.items()):
        perf_matrix = results["performance_matrix"]

        # Build matrix: rows=tested_on, cols=trained_after
        matrix = np.zeros((len(phases), len(phases)))
        matrix[:] = np.nan  # Fill with NaN for phases not yet trained

        for trained_idx, trained_phase in enumerate(phases):
            for tested_idx, tested_phase in enumerate(phases):
                if tested_idx <= trained_idx:  # Can only test on seen tasks
                    key = f"after_{trained_phase}_test_{tested_phase}"
                    if key in perf_matrix:
                        loss = perf_matrix[key]
                        # Convert to performance metric
                        performance = 100 / (1 + loss)
                        matrix[tested_idx, trained_idx] = performance

        # Plot heatmap
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                   xticklabels=[p.capitalize() for p in phases],
                   yticklabels=[p.capitalize() for p in phases],
                   ax=ax, cbar_kws={'label': 'Performance'},
                   mask=np.isnan(matrix))

        ax.set_xlabel('Trained After Phase', fontsize=12)
        ax.set_ylabel('Tested On Task', fontsize=12)
        ax.set_title(f"{model_name.capitalize()} Model\nPerformance Matrix",
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Performance matrix saved to {output_path}")


def plot_forgetting_comparison(results_dict: Dict[str, Dict], output_path: str):
    """
    Bar chart comparing average forgetting across models.

    Args:
        results_dict: Dictionary mapping model names to their results
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = []
    avg_forgetting = []

    for model_name, results in results_dict.items():
        forgetting_scores = results.get("forgetting_scores", {})

        if forgetting_scores:
            # Calculate average forgetting
            forgetting_values = [score["forgetting_percentage"]
                                for score in forgetting_scores.values()]
            avg = np.mean(forgetting_values)

            model_names.append(model_name.capitalize())
            avg_forgetting.append(avg)

    # Create bar chart
    colors = sns.color_palette("husl", len(model_names))
    bars = ax.bar(model_names, avg_forgetting, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=2)

    # Add value labels
    for bar, value in zip(bars, avg_forgetting):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}%',
               ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Average Forgetting (%)', fontsize=14, fontweight='bold')
    ax.set_title('Catastrophic Forgetting Comparison\n(Lower = Better Retention)',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation text
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Excellent (<10%)')
    ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Moderate (<25%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Forgetting comparison saved to {output_path}")


def create_continual_dashboard(results_dict: Dict[str, Dict], output_dir: str):
    """
    Create complete dashboard of continual learning visualizations.

    Args:
        results_dict: Dictionary mapping model names to their results
        output_dir: Directory to save all plots
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Creating continual learning dashboard...")

    # 1. Forgetting curves
    plot_forgetting_curve(
        results_dict,
        os.path.join(output_dir, "forgetting_curves.png")
    )

    # 2. Performance matrices
    plot_performance_matrix(
        results_dict,
        os.path.join(output_dir, "performance_matrices.png")
    )

    # 3. Forgetting heatmaps
    plot_forgetting_heatmap(
        results_dict,
        os.path.join(output_dir, "forgetting_heatmaps.png")
    )

    # 4. Forgetting comparison
    plot_forgetting_comparison(
        results_dict,
        os.path.join(output_dir, "forgetting_comparison.png")
    )

    logger.info(f"Dashboard created successfully in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize continual learning results")
    parser.add_argument("--results_files", nargs="+", required=True,
                       help="Paths to continual_results.json files")
    parser.add_argument("--model_names", nargs="+", required=True,
                       help="Names for each model")
    parser.add_argument("--output_dir", type=str, default="./plots_continual",
                       help="Output directory for plots")

    args = parser.parse_args()

    if len(args.results_files) != len(args.model_names):
        raise ValueError("Number of results files must match number of model names")

    # Load all results
    results_dict = {}
    for results_file, model_name in zip(args.results_files, args.model_names):
        results = load_continual_results(results_file)
        results_dict[model_name] = results

    # Create dashboard
    create_continual_dashboard(results_dict, args.output_dir)
