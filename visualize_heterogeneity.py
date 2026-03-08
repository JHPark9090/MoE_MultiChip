"""
Visualize Circuit MoE heterogeneity clustering results.

Generates publication-quality figures for 3-level clustering:
  Level 1 (Circuit): Gate weight profiles per cluster
  Level 2 (Network): Expert output norms per cluster (PCA + radar)
  Level 3 (ROI):     ROI importance heatmaps per cluster (PCA + brain map)

Usage:
  python visualize_heterogeneity.py --config adhd_3 --model-type classical
  python visualize_heterogeneity.py --config adhd_2 --model-type classical
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from pathlib import Path
import sys
sys.path.insert(0, '.')
from models.yeo17_networks import get_circuit_config


def load_data(config, model_type, version=''):
    """Load heterogeneity analysis results."""
    if version:
        base = Path(f"analysis/heterogeneity_{version}_{model_type}_{config}")
    else:
        base = Path(f"analysis/heterogeneity_{model_type}_{config}")
    npz = np.load(base / "subject_representations.npz", allow_pickle=True)
    with open(base / "heterogeneity_results.json") as f:
        results = json.load(f)
    return npz, results, base


def get_expert_names(config):
    """Get expert names for a given circuit config."""
    circuit_def = get_circuit_config(config)
    return list(circuit_def.keys())


def get_network_names():
    """Return Yeo17 network names in order."""
    from models.yeo17_networks import YEO17_HCP180
    return list(YEO17_HCP180.keys())


# ──────────────────────────────────────────────────────────────
# Level 1: Circuit-level clustering
# ──────────────────────────────────────────────────────────────

def plot_circuit_level(npz, results, expert_names, out_dir):
    """Bar chart of gate weight profiles per cluster."""
    n_clusters = results['n_clusters']
    labels = npz['labels']
    gate_weights = npz['gate_weights']
    cluster_ids = npz['circuit_cluster_ids']

    # ADHD+ subjects only
    adhd_mask = labels == 1
    gate_adhd = gate_weights[adhd_mask]

    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters + 2, 4),
                             sharey=True)
    if n_clusters == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(expert_names)))

    for k in range(n_clusters):
        ax = axes[k]
        mask = cluster_ids == k
        n_subj = mask.sum()
        means = gate_adhd[mask].mean(axis=0)
        stds = gate_adhd[mask].std(axis=0)

        bars = ax.bar(range(len(expert_names)), means, yerr=stds,
                      color=colors, capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(expert_names)))
        ax.set_xticklabels(expert_names, rotation=30, ha='right', fontsize=9)
        ax.set_title(f'Cluster {k} (N={n_subj})', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(0.7, means.max() + stds.max() + 0.05))
        ax.axhline(1.0 / len(expert_names), color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel('Gate Weight', fontsize=11)
    sil = results['silhouettes']['circuit']
    fig.suptitle(f'Circuit-Level Clusters (K={n_clusters}, Silhouette={sil:.3f})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'circuit_level_clusters.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'circuit_level_clusters.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved circuit_level_clusters.pdf/png")


# ──────────────────────────────────────────────────────────────
# Level 2: Network-level clustering
# ──────────────────────────────────────────────────────────────

def plot_network_level(npz, results, expert_names, out_dir):
    """PCA scatter + radar chart of expert output norms per cluster."""
    n_clusters = results['n_clusters']
    labels = npz['labels']
    expert_outputs = npz['expert_outputs']  # (N, K_experts, 64)
    cluster_ids = npz['network_cluster_ids']

    adhd_mask = labels == 1
    expert_adhd = expert_outputs[adhd_mask]
    n_experts = expert_adhd.shape[1]

    # Flatten for PCA: (N_adhd, K*64)
    flat = expert_adhd.reshape(expert_adhd.shape[0], -1)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)

    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    # --- PCA scatter ---
    ax1 = fig.add_subplot(gs[0])
    for k in range(n_clusters):
        mask = cluster_ids == k
        n_subj = mask.sum()
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    c=cluster_colors[k], s=25, alpha=0.6,
                    label=f'Cluster {k} (N={n_subj})', edgecolors='white',
                    linewidths=0.3)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_title('Network-Level PCA', fontsize=12, fontweight='bold')

    # --- Radar chart of expert norms ---
    ax2 = fig.add_subplot(gs[1], polar=True)
    angles = np.linspace(0, 2 * np.pi, n_experts, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for k in range(n_clusters):
        mask = cluster_ids == k
        expert_norms = np.linalg.norm(expert_adhd[mask], axis=2).mean(axis=0)  # (K_experts,)
        values = expert_norms.tolist() + expert_norms[:1].tolist()
        ax2.plot(angles, values, color=cluster_colors[k], linewidth=2,
                 label=f'Cluster {k}')
        ax2.fill(angles, values, color=cluster_colors[k], alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(expert_names, fontsize=9)
    ax2.set_title('Expert Output Norms', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    sil = results['silhouettes']['network']
    fig.suptitle(f'Network-Level Clusters (K={n_clusters}, Silhouette={sil:.3f})',
                 fontsize=13, fontweight='bold', y=1.04)
    fig.savefig(out_dir / 'network_level_clusters.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'network_level_clusters.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved network_level_clusters.pdf/png")


# ──────────────────────────────────────────────────────────────
# Level 3: ROI-level clustering
# ──────────────────────────────────────────────────────────────

def plot_roi_level(npz, results, out_dir):
    """PCA scatter + heatmap of signed ROI scores per cluster."""
    n_clusters = results['n_clusters']
    labels = npz['labels']
    signed_roi = npz['signed_roi_scores']  # (N, 180)
    cluster_ids = npz['roi_cluster_ids']
    network_names = get_network_names()

    adhd_mask = labels == 1
    roi_adhd = signed_roi[adhd_mask]

    # PCA on signed ROI scores
    pca = PCA(n_components=2)
    coords = pca.fit_transform(roi_adhd)

    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # --- PCA scatter ---
    ax1 = fig.add_subplot(gs[0])
    for k in range(n_clusters):
        mask = cluster_ids == k
        n_subj = mask.sum()
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    c=cluster_colors[k], s=25, alpha=0.6,
                    label=f'Cluster {k} (N={n_subj})', edgecolors='white',
                    linewidths=0.3)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_title('ROI-Level PCA', fontsize=12, fontweight='bold')

    # --- Heatmap: mean signed ROI scores per cluster, grouped by network ---
    ax2 = fig.add_subplot(gs[1])

    from models.yeo17_networks import YEO17_HCP180
    net_order = list(YEO17_HCP180.keys())

    # Compute mean signed score per network per cluster
    net_scores = np.zeros((n_clusters, len(net_order)))
    for i, net_name in enumerate(net_order):
        roi_indices = YEO17_HCP180[net_name]
        for k in range(n_clusters):
            mask = cluster_ids == k
            net_scores[k, i] = roi_adhd[mask][:, roi_indices].mean()

    # Scale for visibility
    vmax = np.abs(net_scores).max()
    im = ax2.imshow(net_scores, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(len(net_order)))
    ax2.set_xticklabels(net_order, rotation=55, ha='right', fontsize=8)
    ax2.set_yticks(range(n_clusters))
    ax2.set_yticklabels([f'Cluster {k}\n(N={int((cluster_ids==k).sum())})'
                         for k in range(n_clusters)], fontsize=9)
    ax2.set_title('Mean Signed ROI Scores by Network', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label('Signed Score (+ADHD / -ADHD)', fontsize=9)

    sil = results['silhouettes']['roi']
    fig.suptitle(f'ROI-Level Clusters (K={n_clusters}, Silhouette={sil:.3f})',
                 fontsize=13, fontweight='bold', y=1.04)
    fig.savefig(out_dir / 'roi_level_clusters.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'roi_level_clusters.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved roi_level_clusters.pdf/png")


# ──────────────────────────────────────────────────────────────
# Silhouette sweep
# ──────────────────────────────────────────────────────────────

def plot_silhouette_sweep(results, out_dir):
    """Line plot of silhouette score vs K."""
    sweep = results['silhouette_sweep']
    ks = sorted(int(k) for k in sweep.keys())
    scores = [sweep[str(k)] for k in ks]
    best_k = ks[np.argmax(scores)]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, scores, 'o-', color='#2c3e50', linewidth=2, markersize=8)
    ax.axvline(best_k, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Best K={best_k}')
    used_k = results['n_clusters']
    if used_k != best_k:
        ax.axvline(used_k, color='#3498db', linestyle=':', linewidth=1, alpha=0.7,
                   label=f'Used K={used_k}')
    ax.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Circuit-Level Silhouette Sweep', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(ks)
    plt.tight_layout()
    fig.savefig(out_dir / 'silhouette_sweep.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'silhouette_sweep.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved silhouette_sweep.pdf/png")


# ──────────────────────────────────────────────────────────────
# Cross-level concordance
# ──────────────────────────────────────────────────────────────

def plot_cross_level_ari(results, out_dir):
    """Bar chart of Adjusted Rand Index between levels."""
    ari = results['cross_level_ari']
    pairs = ['Circuit vs\nNetwork', 'Circuit vs\nROI', 'Network vs\nROI']
    values = [ari['circuit_vs_network'], ari['circuit_vs_roi'], ari['network_vs_roi']]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(range(3), values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(pairs, fontsize=10)
    ax.set_ylabel('Adjusted Rand Index', fontsize=11)
    ax.set_title('Cross-Level Concordance', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_dir / 'cross_level_ari.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'cross_level_ari.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved cross_level_ari.pdf/png")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=['adhd_3', 'adhd_2'])
    parser.add_argument('--model-type', required=True, choices=['classical', 'quantum'])
    parser.add_argument('--version', default='', help='Version prefix (e.g., "v5")')
    args = parser.parse_args()

    # For quantum model type, adjust the directory suffix
    model_suffix = args.model_type
    if args.model_type == 'quantum':
        model_suffix = 'quantum_8q_d3'

    print(f"\nVisualizing heterogeneity: {model_suffix} {args.config}")
    print("=" * 60)

    npz, results, base_dir = load_data(args.config, model_suffix, args.version)
    expert_names = get_expert_names(args.config)

    out_dir = base_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_silhouette_sweep(results, out_dir)
    plot_circuit_level(npz, results, expert_names, out_dir)
    plot_network_level(npz, results, expert_names, out_dir)
    plot_roi_level(npz, results, out_dir)
    plot_cross_level_ari(results, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
