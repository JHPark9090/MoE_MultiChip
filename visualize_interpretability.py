"""
Visualize Circuit MoE interpretability analysis results.

Generates publication-quality figures for 3-level interpretability:
  1. Gate weights: ADHD+ vs ADHD- comparison per circuit
  2. Circuit & Network saliency: Absolute + signed bar charts
  3. ROI saliency: Top ROIs ranked by importance, colored by direction
  4. Input weight analysis: Expert-to-network heatmap

Usage:
  python visualize_interpretability.py --config adhd_3 --model-type classical --version v5
  python visualize_interpretability.py --config adhd_2 --model-type quantum --version v5
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
sys.path.insert(0, '.')
from models.yeo17_networks import get_circuit_config


def load_data(config, model_type, version=''):
    """Load interpretability analysis results."""
    if version:
        base = Path(f"analysis/{version}_{model_type}_{config}")
    else:
        base = Path(f"analysis/{model_type}_{config}")
    with open(base / "results.json") as f:
        results = json.load(f)
    return results, base


def get_expert_names(config):
    circuit_def = get_circuit_config(config)
    return list(circuit_def.keys())


# ──────────────────────────────────────────────────────────────
# Plot 1: Gate Weight Comparison (ADHD+ vs ADHD-)
# ──────────────────────────────────────────────────────────────

def plot_gate_weights(results, expert_names, out_dir):
    """Grouped bar chart: ADHD+ vs ADHD- gate weights per circuit with p-values."""
    stats = results['gate_weight_stats']
    n = len(expert_names)

    pos_means = [stats[e]['adhd_pos_mean'] for e in expert_names]
    neg_means = [stats[e]['adhd_neg_mean'] for e in expert_names]
    pos_stds = [stats[e]['adhd_pos_std'] for e in expert_names]
    neg_stds = [stats[e]['adhd_neg_std'] for e in expert_names]
    p_values = [stats[e]['p_value'] for e in expert_names]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 4.5))
    bars1 = ax.bar(x - width/2, pos_means, width, yerr=pos_stds,
                   label='ADHD+', color='#e74c3c', capsize=4,
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, neg_means, width, yerr=neg_stds,
                   label='ADHD-', color='#3498db', capsize=4,
                   edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add significance stars
    y_max = max(max(p + s for p, s in zip(pos_means, pos_stds)),
                max(p + s for p, s in zip(neg_means, neg_stds)))
    for i, p in enumerate(p_values):
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.text(i, y_max + 0.003, star, ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='#e74c3c' if p < 0.05 else 'gray')

    ax.set_xticks(x)
    ax.set_xticklabels(expert_names, fontsize=11)
    ax.set_ylabel('Gate Weight', fontsize=12)
    ax.set_title('Gate Weights: ADHD+ vs ADHD-', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(1.0 / n, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.5, label='Uniform')

    plt.tight_layout()
    fig.savefig(out_dir / 'gate_weights_comparison.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'gate_weights_comparison.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved gate_weights_comparison.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 2: Circuit-Level Saliency (Absolute + Signed)
# ──────────────────────────────────────────────────────────────

def plot_circuit_saliency(results, expert_names, out_dir):
    """Side-by-side bar charts: absolute saliency and signed saliency per circuit."""
    sal = results['circuit_saliency']
    n = len(expert_names)

    abs_vals = [sal[e]['all_mean'] for e in expert_names]
    signed_vals = [sal[e]['signed_all'] for e in expert_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, n * 2.5), 4.5))

    colors_abs = plt.cm.Set2(np.linspace(0, 1, n))

    # Absolute saliency
    bars = ax1.bar(range(n), abs_vals, color=colors_abs,
                   edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(expert_names, rotation=20, ha='right', fontsize=10)
    ax1.set_ylabel('Absolute Saliency', fontsize=11)
    ax1.set_title('Circuit Absolute Saliency', fontsize=12, fontweight='bold')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    # Signed saliency (diverging)
    colors_signed = ['#e74c3c' if v > 0 else '#3498db' for v in signed_vals]
    ax2.bar(range(n), signed_vals, color=colors_signed,
            edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(expert_names, rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('Signed Saliency', fontsize=11)
    ax2.set_title('Circuit Signed Saliency (+ADHD / -ADHD)', fontsize=12, fontweight='bold')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    # Add +ADHD/-ADHD labels
    for i, v in enumerate(signed_vals):
        label = '+ADHD' if v > 0 else '-ADHD'
        ax2.text(i, v + (abs(v) * 0.1 if v > 0 else -abs(v) * 0.1),
                 label, ha='center', va='bottom' if v > 0 else 'top',
                 fontsize=8, fontweight='bold', color=colors_signed[i])

    plt.tight_layout()
    fig.savefig(out_dir / 'circuit_saliency.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'circuit_saliency.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved circuit_saliency.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 3: Network-Level Saliency (Absolute + Signed)
# ──────────────────────────────────────────────────────────────

def plot_network_saliency(results, out_dir):
    """Horizontal bar charts: absolute and signed saliency per Yeo-17 network."""
    sal = results['network_saliency']
    networks = list(sal.keys())
    n = len(networks)

    abs_vals = [sal[net]['all_mean'] for net in networks]
    signed_vals = [sal[net]['signed_all'] for net in networks]

    # Sort by absolute saliency for better readability
    sort_idx = np.argsort(abs_vals)
    networks_sorted = [networks[i] for i in sort_idx]
    abs_sorted = [abs_vals[i] for i in sort_idx]
    signed_sorted = [signed_vals[i] for i in sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n * 0.35)),
                                    sharey=True)

    # Absolute saliency (horizontal bars, sorted)
    colors_abs = plt.cm.viridis(np.linspace(0.2, 0.9, n))
    ax1.barh(range(n), abs_sorted, color=colors_abs,
             edgecolor='black', linewidth=0.3)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(networks_sorted, fontsize=9)
    ax1.set_xlabel('Absolute Saliency', fontsize=11)
    ax1.set_title('Network Absolute Saliency', fontsize=12, fontweight='bold')
    ax1.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

    # Signed saliency (diverging horizontal bars, same sort order)
    colors_signed = ['#e74c3c' if v > 0 else '#3498db' for v in signed_sorted]
    ax2.barh(range(n), signed_sorted, color=colors_signed,
             edgecolor='black', linewidth=0.3)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Signed Saliency', fontsize=11)
    ax2.set_title('Network Signed Saliency (+ADHD / -ADHD)', fontsize=12,
                  fontweight='bold')
    ax2.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

    # Add direction labels
    for i, v in enumerate(signed_sorted):
        label = '+' if v > 0 else '-'
        ax2.text(v, i, f' {label}', ha='left' if v > 0 else 'right',
                 va='center', fontsize=8, fontweight='bold',
                 color=colors_signed[i])

    plt.tight_layout()
    fig.savefig(out_dir / 'network_saliency.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'network_saliency.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved network_saliency.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 4: Top ROI Saliency Rankings
# ──────────────────────────────────────────────────────────────

def plot_roi_saliency(results, out_dir, top_n=20):
    """Top-N ROIs by absolute saliency, colored by signed direction."""
    rois = results['roi_saliency_top50'][:top_n]

    regions = [f"{r['region']} ({r['network']})" for r in rois]
    abs_vals = [r['saliency_all'] for r in rois]
    signed_vals = [r['signed_all'] for r in rois]

    # Reverse for horizontal bar (top at top)
    regions = regions[::-1]
    abs_vals = abs_vals[::-1]
    signed_vals = signed_vals[::-1]

    colors = ['#e74c3c' if s > 0 else '#3498db' for s in signed_vals]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(range(len(regions)), abs_vals, color=colors,
            edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=9)
    ax.set_xlabel('Absolute Saliency', fontsize=11)
    ax.set_title(f'Top {top_n} ROIs by Saliency (Red=+ADHD, Blue=-ADHD)',
                 fontsize=12, fontweight='bold')
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='+ADHD'),
                       Patch(facecolor='#3498db', label='-ADHD')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / 'roi_saliency_top20.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'roi_saliency_top20.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved roi_saliency_top20.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 5: Signed ROI Rankings (+ADHD and -ADHD)
# ──────────────────────────────────────────────────────────────

def plot_signed_roi_rankings(results, out_dir):
    """Side-by-side: top +ADHD ROIs and top -ADHD ROIs."""
    pos_rois = results['roi_signed_top20_positive'][:10]
    neg_rois = results['roi_signed_top20_negative'][:10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # +ADHD ROIs
    if pos_rois:
        names_p = [f"{r['region']} ({r['network']})" for r in pos_rois][::-1]
        vals_p = [r['signed_all'] for r in pos_rois][::-1]
        ax1.barh(range(len(names_p)), vals_p, color='#e74c3c',
                 edgecolor='black', linewidth=0.3)
        ax1.set_yticks(range(len(names_p)))
        ax1.set_yticklabels(names_p, fontsize=9)
    ax1.set_xlabel('Signed Saliency', fontsize=11)
    ax1.set_title('Top 10 +ADHD ROIs', fontsize=12, fontweight='bold',
                  color='#e74c3c')
    ax1.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

    # -ADHD ROIs
    if neg_rois:
        names_n = [f"{r['region']} ({r['network']})" for r in neg_rois][::-1]
        vals_n = [r['signed_all'] for r in neg_rois][::-1]
        ax2.barh(range(len(names_n)), vals_n, color='#3498db',
                 edgecolor='black', linewidth=0.3)
        ax2.set_yticks(range(len(names_n)))
        ax2.set_yticklabels(names_n, fontsize=9)
    ax2.set_xlabel('Signed Saliency', fontsize=11)
    ax2.set_title('Top 10 -ADHD ROIs', fontsize=12, fontweight='bold',
                  color='#3498db')
    ax2.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

    plt.tight_layout()
    fig.savefig(out_dir / 'signed_roi_rankings.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'signed_roi_rankings.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved signed_roi_rankings.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 6: Expert Input Weight Heatmap
# ──────────────────────────────────────────────────────────────

def plot_input_weight_heatmap(results, expert_names, out_dir):
    """Heatmap: expert × network showing learned input projection weights."""
    iwa = results['input_weight_analysis']
    from models.yeo17_networks import YEO17_HCP180
    network_names = list(YEO17_HCP180.keys())

    # Build matrix: experts × networks
    mat = np.zeros((len(expert_names), len(network_names)))
    for i, exp in enumerate(expert_names):
        net_summary = iwa[exp]['network_summary']
        for j, net in enumerate(network_names):
            if net in net_summary:
                entry = net_summary[net]
                mat[i, j] = entry['mean_weight_norm'] if isinstance(entry, dict) else entry

    fig, ax = plt.subplots(figsize=(max(10, len(network_names) * 0.7),
                                    max(3, len(expert_names) * 0.8 + 1)))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(network_names)))
    ax.set_xticklabels(network_names, rotation=55, ha='right', fontsize=9)
    ax.set_yticks(range(len(expert_names)))
    ax.set_yticklabels(expert_names, fontsize=11)
    ax.set_title('Expert Input Projection Weights by Network', fontsize=13,
                 fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean |Weight|', fontsize=10)

    # Highlight diagonal blocks (assigned networks should have highest weights)
    for i in range(len(expert_names)):
        max_j = np.argmax(mat[i])
        ax.add_patch(plt.Rectangle((max_j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='black',
                                    linewidth=2))

    plt.tight_layout()
    fig.savefig(out_dir / 'input_weight_heatmap.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'input_weight_heatmap.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved input_weight_heatmap.pdf/png")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Circuit MoE interpretability results')
    parser.add_argument('--config', required=True, choices=['adhd_3', 'adhd_2'])
    parser.add_argument('--model-type', required=True,
                        choices=['classical', 'quantum'])
    parser.add_argument('--version', default='',
                        help='Version prefix (e.g., "v5")')
    args = parser.parse_args()

    # For quantum model type, adjust the directory suffix
    model_suffix = args.model_type
    if args.model_type == 'quantum':
        model_suffix = 'quantum_8q_d3'

    print(f"\nVisualizing interpretability: {model_suffix} {args.config}")
    print("=" * 60)

    results, base_dir = load_data(args.config, model_suffix, args.version)
    expert_names = get_expert_names(args.config)

    out_dir = base_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_gate_weights(results, expert_names, out_dir)
    plot_circuit_saliency(results, expert_names, out_dir)
    plot_network_saliency(results, out_dir)
    plot_roi_saliency(results, out_dir)
    plot_signed_roi_rankings(results, out_dir)
    plot_input_weight_heatmap(results, expert_names, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
