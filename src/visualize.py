"""
ValuePriorityBench: Visualization Module
Generates figures for the paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Color scheme
COLORS = {
    'safety': '#E74C3C',      # Red
    'honesty': '#3498DB',     # Blue
    'helpfulness': '#2ECC71', # Green
    'compliance': '#9B59B6',  # Purple
}

MODEL_COLORS = {
    'claude': '#D4A574',
    'gpt': '#74B9A7',
    'gemini': '#7BA3D4',
    'deepseek': '#D47B7B',
    'kimi': '#B97BD4',
}

VALUE_LABELS = {
    'safety': 'Safety',
    'honesty': 'Honesty',
    'helpfulness': 'Helpfulness',
    'compliance': 'Compliance',
}

MODEL_LABELS = {
    'claude': 'Claude',
    'gpt': 'GPT',
    'gemini': 'Gemini',
    'deepseek': 'DeepSeek',
    'kimi': 'Kimi',
}

def load_results(results_path: str) -> Dict:
    """Load inference results."""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_priority_strengths(results: Dict, output_path: str):
    """
    Plot priority strength estimates with confidence intervals for each model.
    """
    models = list(results.keys())
    n_models = len(models)
    values = ['safety', 'honesty', 'helpfulness', 'compliance']

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_id in zip(axes, models):
        estimates = results[model_id]['priority_estimates']
        est_dict = {e['value']: e for e in estimates}

        y_pos = np.arange(len(values))
        means = [est_dict[v]['mean'] for v in values]
        errors_low = [est_dict[v]['mean'] - est_dict[v]['hdi_low'] for v in values]
        errors_high = [est_dict[v]['hdi_high'] - est_dict[v]['mean'] for v in values]

        colors = [COLORS[v] for v in values]

        ax.barh(y_pos, means, xerr=[errors_low, errors_high],
                color=colors, alpha=0.7, capsize=3, ecolor='gray')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([VALUE_LABELS[v] for v in values])
        ax.set_xlabel('Priority Strength')
        ax.set_title(MODEL_LABELS.get(model_id, model_id))
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_pairwise_heatmap(results: Dict, output_path: str):
    """
    Plot heatmap of P(value_i > value_j) for each model.
    """
    models = list(results.keys())
    n_models = len(models)
    values = ['safety', 'honesty', 'helpfulness', 'compliance']
    n_values = len(values)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5))
    if n_models == 1:
        axes = [axes]

    for ax, model_id in zip(axes, models):
        # Create probability matrix
        prob_matrix = np.zeros((n_values, n_values))

        pairwise = results[model_id]['pairwise_probabilities']
        for p in pairwise:
            i = values.index(p['value_i'])
            j = values.index(p['value_j'])
            prob_matrix[i, j] = p['probability']
            prob_matrix[j, i] = 1 - p['probability']

        # Diagonal = 0.5 (equal to self)
        np.fill_diagonal(prob_matrix, 0.5)

        # Plot heatmap
        im = ax.imshow(prob_matrix, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')

        # Add text annotations
        for i in range(n_values):
            for j in range(n_values):
                text = f'{prob_matrix[i, j]:.2f}'
                color = 'white' if abs(prob_matrix[i, j] - 0.5) > 0.25 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

        ax.set_xticks(range(n_values))
        ax.set_yticks(range(n_values))
        ax.set_xticklabels([VALUE_LABELS[v] for v in values], rotation=45, ha='right')
        ax.set_yticklabels([VALUE_LABELS[v] for v in values])
        ax.set_title(MODEL_LABELS.get(model_id, model_id))
        ax.set_xlabel('P(row > column)')

    # Add colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Probability')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_priority_dag(results: Dict, output_path: str, threshold: float = 0.75):
    """
    Plot Priority DAG showing significant dominance relationships.
    """
    models = list(results.keys())
    n_models = len(models)
    values = ['safety', 'honesty', 'helpfulness', 'compliance']

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model_id in zip(axes, models):
        # Get ordering and create positions
        ordering = results[model_id]['priority_ordering']
        n = len(ordering)

        # Position nodes vertically by rank
        positions = {v: (0.5, (n - ordering.index(v) - 0.5) / n) for v in ordering}

        # Draw nodes
        for v, (x, y) in positions.items():
            circle = plt.Circle((x, y), 0.08, color=COLORS[v], alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, VALUE_LABELS[v][:3], ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')

        # Draw edges for significant relationships
        pairwise = results[model_id]['pairwise_probabilities']
        for p in pairwise:
            prob = p['probability']
            if prob > threshold:
                # Draw arrow from higher to lower
                start = positions[p['value_i']]
                end = positions[p['value_j']]
            elif prob < (1 - threshold):
                start = positions[p['value_j']]
                end = positions[p['value_i']]
            else:
                continue

            # Offset arrow endpoints
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
                start_adj = (start[0] + dx*0.09, start[1] + dy*0.09)
                end_adj = (end[0] - dx*0.09, end[1] - dy*0.09)

                ax.annotate('', xy=end_adj, xytext=start_adj,
                           arrowprops=dict(arrowstyle='->', color='gray',
                                          lw=1.5, alpha=0.7))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(MODEL_LABELS.get(model_id, model_id))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_pas_comparison(results: Dict, output_path: str):
    """
    Plot PAS comparison across models.
    """
    from bayesian_inference import compute_priority_alignment_score, compute_weighted_pas

    claude_declared = ["safety", "honesty", "compliance", "helpfulness"]

    models = list(results.keys())
    pas_scores = []
    weighted_pas_scores = []

    for model_id in models:
        ordering = results[model_id]['priority_ordering']
        pas, _ = compute_priority_alignment_score(ordering, claude_declared)
        weighted_pas = compute_weighted_pas(ordering, claude_declared)
        pas_scores.append(pas)
        weighted_pas_scores.append(weighted_pas)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, pas_scores, width, label='PAS (Kendall τ)',
                   color=[MODEL_COLORS.get(m, 'gray') for m in models], alpha=0.7)
    bars2 = ax.bar(x + width/2, weighted_pas_scores, width, label='Weighted PAS',
                   color=[MODEL_COLORS.get(m, 'gray') for m in models], alpha=0.4,
                   hatch='//')

    ax.set_ylabel('Alignment Score')
    ax.set_xlabel('Model')
    ax.set_title('Priority Alignment with Claude Constitution')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Add value labels on bars
    for bar, score in zip(bars1, pas_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_cross_model_similarity(results: Dict, output_path: str):
    """
    Plot similarity matrix between models' priority orderings.
    """
    from scipy.stats import kendalltau

    models = list(results.keys())
    n_models = len(models)

    # Compute similarity matrix
    similarity = np.zeros((n_models, n_models))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            ord1 = results[m1]['priority_ordering']
            ord2 = results[m2]['priority_ordering']

            # Kendall tau correlation
            ranks1 = [ord1.index(v) for v in ord1]
            ranks2 = [ord2.index(v) for v in ord1]
            tau, _ = kendalltau(ranks1, ranks2)
            similarity[i, j] = tau

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)

    # Add annotations
    for i in range(n_models):
        for j in range(n_models):
            text = f'{similarity[i, j]:.2f}'
            color = 'white' if abs(similarity[i, j]) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=45, ha='right')
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_title('Cross-Model Priority Similarity (Kendall τ)')

    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_choice_distribution(parsed_path: str, output_path: str):
    """
    Plot distribution of value choices across scenarios for each model.
    """
    with open(parsed_path, 'r') as f:
        parsed = json.load(f)

    # Aggregate choices by model and value
    choice_counts = defaultdict(lambda: defaultdict(int))
    for p in parsed:
        if p['chosen_value']:
            choice_counts[p['model_id']][p['chosen_value']] += 1

    models = list(choice_counts.keys())
    values = ['safety', 'honesty', 'helpfulness', 'compliance']

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.6

    bottom = np.zeros(len(models))

    for value in values:
        counts = [choice_counts[m].get(value, 0) for m in models]
        ax.bar(x, counts, width, label=VALUE_LABELS[value], bottom=bottom,
               color=COLORS[value], alpha=0.8)
        bottom += counts

    ax.set_ylabel('Number of Choices')
    ax.set_xlabel('Model')
    ax.set_title('Distribution of Value Choices Across All Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_all_figures(
    results_path: str,
    parsed_path: str,
    figures_dir: str
):
    """Generate all figures for the paper."""
    results = load_results(results_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(exist_ok=True)

    print("\nGenerating figures...")

    # 1. Priority strength estimates
    plot_priority_strengths(
        results,
        str(figures_dir / "fig1_priority_strengths.png")
    )

    # 2. Pairwise probability heatmap
    plot_pairwise_heatmap(
        results,
        str(figures_dir / "fig2_pairwise_heatmap.png")
    )

    # 3. Priority DAG
    plot_priority_dag(
        results,
        str(figures_dir / "fig3_priority_dag.png")
    )

    # 4. PAS comparison
    plot_pas_comparison(
        results,
        str(figures_dir / "fig4_pas_comparison.png")
    )

    # 5. Cross-model similarity
    plot_cross_model_similarity(
        results,
        str(figures_dir / "fig5_cross_model_similarity.png")
    )

    # 6. Choice distribution
    if Path(parsed_path).exists():
        plot_choice_distribution(
            parsed_path,
            str(figures_dir / "fig6_choice_distribution.png")
        )

    print("\nAll figures generated!")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    results_path = base_dir / "results" / "inference_results.json"
    parsed_path = base_dir / "data" / "responses" / "parsed_choices.json"
    figures_dir = base_dir / "figures"

    generate_all_figures(
        str(results_path),
        str(parsed_path),
        str(figures_dir)
    )
