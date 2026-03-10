"""
Visualize per-expert gradient saliency analysis results.

Generates publication-quality figures for per-expert saliency:
  1. Intra-circuit network saliency heatmap (experts x networks)
  2. Per-expert top ROI bar charts (abs + signed)
  3. Cross-expert network hierarchy comparison
  4. ADHD+ vs ADHD- volcano plot per expert
  5. Cross-model comparison (classical vs quantum, 2e vs 4e)

Usage:
  # Single model:
  python visualize_expert_saliency.py \
      --input-dir=analysis/expert_saliency_v5_classical_adhd_3

  # Cross-model comparison (all 4 models):
  python visualize_expert_saliency.py --cross-model
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
sys.path.insert(0, '.')
from models.yeo17_networks import get_circuit_config


# Color palette
C_POS = '#e74c3c'   # +ADHD (red)
C_NEG = '#3498db'   # -ADHD (blue)
C_NS = '#95a5a6'    # not significant (gray)
EXPERT_COLORS = ['#2ecc71', '#e67e22', '#9b59b6', '#1abc9c']  # green, orange, purple, teal


def load_results(input_dir):
    """Load per-expert saliency results JSON."""
    p = Path(input_dir)
    with open(p / "expert_saliency_results.json") as f:
        return json.load(f), p


# ──────────────────────────────────────────────────────────────
# Plot 1: Intra-Circuit Network Saliency Heatmap
# ──────────────────────────────────────────────────────────────

def plot_network_saliency_heatmap(results, out_dir):
    """Heatmap: experts × networks showing absolute & signed saliency."""
    config_name = results['circuit_config']
    config = get_circuit_config(config_name)
    expert_names = list(config.keys())

    # Collect all networks across all experts
    all_networks = []
    for exp in expert_names:
        if exp in results:
            for net in results[exp]['network_analysis']:
                if net not in all_networks:
                    all_networks.append(net)

    n_exp = len(expert_names)
    n_net = len(all_networks)

    # Build matrices
    abs_mat = np.full((n_exp, n_net), np.nan)
    signed_mat = np.full((n_exp, n_net), np.nan)
    p_mat = np.full((n_exp, n_net), np.nan)

    for i, exp in enumerate(expert_names):
        if exp not in results:
            continue
        na = results[exp]['network_analysis']
        for j, net in enumerate(all_networks):
            if net in na:
                abs_mat[i, j] = na[net]['abs_mean_all']
                signed_mat[i, j] = na[net]['signed_mean_all']
                p_mat[i, j] = na[net]['signed_p_value']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n_net * 1.2), max(3, n_exp * 0.9 + 1.5)))

    # Absolute saliency heatmap
    im1 = ax1.imshow(abs_mat, aspect='auto', cmap='YlOrRd')
    ax1.set_xticks(range(n_net))
    ax1.set_xticklabels(all_networks, rotation=50, ha='right', fontsize=8)
    ax1.set_yticks(range(n_exp))
    ax1.set_yticklabels(expert_names, fontsize=10)
    ax1.set_title('Absolute Saliency', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.ax.tick_params(labelsize=8)

    # Mark NaN cells (networks not in this expert's circuit)
    for i in range(n_exp):
        for j in range(n_net):
            if np.isnan(abs_mat[i, j]):
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, facecolor='#f0f0f0',
                                            edgecolor='lightgray', linewidth=0.5))
                ax1.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')

    # Signed saliency heatmap (diverging)
    vmax = np.nanmax(np.abs(signed_mat))
    im2 = ax2.imshow(signed_mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(n_net))
    ax2.set_xticklabels(all_networks, rotation=50, ha='right', fontsize=8)
    ax2.set_yticks(range(n_exp))
    ax2.set_yticklabels(expert_names, fontsize=10)
    ax2.set_title('Signed Saliency (+ADHD / -ADHD)', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.ax.tick_params(labelsize=8)

    # Mark significant cells and NaN cells
    for i in range(n_exp):
        for j in range(n_net):
            if np.isnan(signed_mat[i, j]):
                ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, facecolor='#f0f0f0',
                                            edgecolor='lightgray', linewidth=0.5))
                ax2.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
            elif not np.isnan(p_mat[i, j]) and p_mat[i, j] < 0.05:
                ax2.text(j, i, '*', ha='center', va='center', fontsize=14,
                         fontweight='bold', color='black')

    fig.suptitle(f'Per-Expert Network Saliency — {results["model_type"]} {config_name}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'network_saliency_heatmap.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'network_saliency_heatmap.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved network_saliency_heatmap.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 2: Per-Expert Top ROI Bar Charts
# ──────────────────────────────────────────────────────────────

def plot_expert_top_rois(results, out_dir, top_n=10):
    """One subplot per expert: top ROIs by absolute saliency, colored by direction."""
    config_name = results['circuit_config']
    config = get_circuit_config(config_name)
    expert_names = list(config.keys())
    active_experts = [e for e in expert_names if e in results]

    n_exp = len(active_experts)
    fig, axes = plt.subplots(1, n_exp, figsize=(5.5 * n_exp, max(5, top_n * 0.4)),
                              sharey=False)
    if n_exp == 1:
        axes = [axes]

    for idx, exp in enumerate(active_experts):
        ax = axes[idx]
        rois = results[exp]['roi_analysis_top20'][:top_n]

        names = [f"{r['region']} ({r['network']})" for r in rois][::-1]
        abs_vals = [r['abs_mean_all'] for r in rois][::-1]
        signed_vals = [r['signed_mean_all'] for r in rois][::-1]
        p_vals = [r['signed_p_value'] for r in rois][::-1]

        colors = []
        for s, p in zip(signed_vals, p_vals):
            if p < 0.05:
                colors.append(C_POS if s > 0 else C_NEG)
            else:
                colors.append('#ff9999' if s > 0 else '#99ccff')  # lighter for n.s.

        ax.barh(range(len(names)), abs_vals, color=colors,
                edgecolor='black', linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Absolute Saliency', fontsize=9)
        ax.set_title(f'{exp}\n({results[exp]["n_rois"]} ROIs)',
                     fontsize=11, fontweight='bold', color=EXPERT_COLORS[idx % len(EXPERT_COLORS)])
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

        # Add significance markers
        for i, (p, v) in enumerate(zip(p_vals, abs_vals)):
            if p < 0.05:
                ax.text(v, i, ' *', ha='left', va='center', fontsize=10,
                        fontweight='bold', color='black')

    legend_elements = [
        Patch(facecolor=C_POS, label='+ADHD (p<0.05)'),
        Patch(facecolor='#ff9999', label='+ADHD (n.s.)'),
        Patch(facecolor=C_NEG, label='-ADHD (p<0.05)'),
        Patch(facecolor='#99ccff', label='-ADHD (n.s.)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f'Top {top_n} ROIs per Expert — {results["model_type"]} {config_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / 'expert_top_rois.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'expert_top_rois.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved expert_top_rois.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 3: Intra-Circuit Feature Hierarchy (bar per expert)
# ──────────────────────────────────────────────────────────────

def plot_feature_hierarchy(results, out_dir):
    """Grouped bar chart: network saliency within each expert, ranked."""
    config_name = results['circuit_config']
    config = get_circuit_config(config_name)
    expert_names = [e for e in config.keys() if e in results]
    n_exp = len(expert_names)

    fig, axes = plt.subplots(1, n_exp, figsize=(4.5 * n_exp, 4), sharey=False)
    if n_exp == 1:
        axes = [axes]

    for idx, exp in enumerate(expert_names):
        ax = axes[idx]
        na = results[exp]['network_analysis']

        # Sort by absolute saliency
        sorted_nets = sorted(na.items(), key=lambda x: x[1]['abs_mean_all'], reverse=True)
        net_names = [n for n, _ in sorted_nets]
        abs_vals = [v['abs_mean_all'] for _, v in sorted_nets]
        signed_vals = [v['signed_mean_all'] for _, v in sorted_nets]
        p_vals = [v['signed_p_value'] for _, v in sorted_nets]

        colors = [C_POS if s > 0 else C_NEG for s in signed_vals]
        edge_colors = ['black' if p < 0.05 else 'gray' for p in p_vals]
        line_widths = [1.5 if p < 0.05 else 0.5 for p in p_vals]

        bars = ax.bar(range(len(net_names)), abs_vals, color=colors,
                      edgecolor=edge_colors, linewidth=line_widths)
        ax.set_xticks(range(len(net_names)))
        ax.set_xticklabels(net_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Absolute Saliency', fontsize=9)
        ax.set_title(f'{exp}', fontsize=11, fontweight='bold',
                     color=EXPERT_COLORS[idx % len(EXPERT_COLORS)])
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

        # Significance stars
        for i, p in enumerate(p_vals):
            if p < 0.05:
                star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                ax.text(i, abs_vals[i], star, ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        # Direction labels
        for i, (s, v) in enumerate(zip(signed_vals, abs_vals)):
            label = '+' if s > 0 else '-'
            ax.text(i, v * 0.5, label, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white', alpha=0.8)

    fig.suptitle(f'Intra-Circuit Feature Hierarchy — {results["model_type"]} {config_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / 'feature_hierarchy.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'feature_hierarchy.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved feature_hierarchy.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 4: Volcano Plot (signed saliency vs -log10 p-value)
# ──────────────────────────────────────────────────────────────

def plot_volcano(results, out_dir):
    """Volcano plot per expert: signed saliency (x) vs -log10(p) (y) for each ROI."""
    config_name = results['circuit_config']
    config = get_circuit_config(config_name)
    expert_names = [e for e in config.keys() if e in results]
    n_exp = len(expert_names)

    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5), sharey=True)
    if n_exp == 1:
        axes = [axes]

    for idx, exp in enumerate(expert_names):
        ax = axes[idx]
        rois = results[exp]['roi_analysis_top20']

        # Use all ROIs from JSON (top 20)
        signed = [r['signed_mean_all'] for r in rois]
        pvals = [max(r['signed_p_value'], 1e-10) for r in rois]
        neg_log_p = [-np.log10(p) for p in pvals]
        names = [r['region'] for r in rois]

        colors = []
        for s, p in zip(signed, pvals):
            if p < 0.05 and s > 0:
                colors.append(C_POS)
            elif p < 0.05 and s < 0:
                colors.append(C_NEG)
            else:
                colors.append(C_NS)

        ax.scatter(signed, neg_log_p, c=colors, s=50, alpha=0.8,
                   edgecolors='black', linewidth=0.3)

        # Label significant ROIs
        for s, nlp, name, p in zip(signed, neg_log_p, names, pvals):
            if p < 0.05:
                ax.annotate(name, (s, nlp), fontsize=7, ha='center',
                           va='bottom', textcoords='offset points',
                           xytext=(0, 5))

        ax.axhline(-np.log10(0.05), color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Signed Saliency', fontsize=10)
        if idx == 0:
            ax.set_ylabel('-log10(p-value)', fontsize=10)
        ax.set_title(f'{exp}', fontsize=11, fontweight='bold',
                     color=EXPERT_COLORS[idx % len(EXPERT_COLORS)])
        ax.text(0.02, 0.98, f'n={results[exp]["n_rois"]} ROIs',
                transform=ax.transAxes, fontsize=8, va='top')

    fig.suptitle(f'Volcano Plot — Per-Expert ROI Saliency\n{results["model_type"]} {config_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / 'volcano_plot.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'volcano_plot.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved volcano_plot.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 5: +ADHD and -ADHD Top ROIs per Expert (signed)
# ──────────────────────────────────────────────────────────────

def plot_signed_rois_per_expert(results, out_dir, top_n=5):
    """Side-by-side: top +ADHD and -ADHD ROIs for each expert."""
    config_name = results['circuit_config']
    config = get_circuit_config(config_name)
    expert_names = [e for e in config.keys() if e in results]
    n_exp = len(expert_names)

    fig, axes = plt.subplots(n_exp, 2, figsize=(14, 3.5 * n_exp))
    if n_exp == 1:
        axes = axes.reshape(1, -1)

    for idx, exp in enumerate(expert_names):
        pos_rois = results[exp].get('roi_signed_top10_positive', [])[:top_n]
        neg_rois = results[exp].get('roi_signed_top10_negative', [])[:top_n]

        # +ADHD
        ax_pos = axes[idx, 0]
        if pos_rois:
            names_p = [f"{r['region']} ({r['network']})" for r in pos_rois][::-1]
            vals_p = [r['signed_mean_all'] for r in pos_rois][::-1]
            p_vals_p = [r['signed_p_value'] for r in pos_rois][::-1]
            colors_p = [C_POS if p < 0.05 else '#ff9999' for p in p_vals_p]
            ax_pos.barh(range(len(names_p)), vals_p, color=colors_p,
                        edgecolor='black', linewidth=0.3)
            ax_pos.set_yticks(range(len(names_p)))
            ax_pos.set_yticklabels(names_p, fontsize=8)
            for i, p in enumerate(p_vals_p):
                if p < 0.05:
                    ax_pos.text(vals_p[i], i, ' *', ha='left', va='center',
                                fontsize=10, fontweight='bold')
        ax_pos.set_xlabel('Signed Saliency', fontsize=9)
        ax_pos.set_title(f'{exp} — Top {top_n} +ADHD', fontsize=10,
                         fontweight='bold', color=C_POS)
        ax_pos.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

        # -ADHD
        ax_neg = axes[idx, 1]
        if neg_rois:
            names_n = [f"{r['region']} ({r['network']})" for r in neg_rois][::-1]
            vals_n = [r['signed_mean_all'] for r in neg_rois][::-1]
            p_vals_n = [r['signed_p_value'] for r in neg_rois][::-1]
            colors_n = [C_NEG if p < 0.05 else '#99ccff' for p in p_vals_n]
            ax_neg.barh(range(len(names_n)), vals_n, color=colors_n,
                        edgecolor='black', linewidth=0.3)
            ax_neg.set_yticks(range(len(names_n)))
            ax_neg.set_yticklabels(names_n, fontsize=8)
            for i, p in enumerate(p_vals_n):
                if p < 0.05:
                    ax_neg.text(vals_n[i], i, ' *', ha='right', va='center',
                                fontsize=10, fontweight='bold')
        ax_neg.set_xlabel('Signed Saliency', fontsize=9)
        ax_neg.set_title(f'{exp} — Top {top_n} -ADHD', fontsize=10,
                         fontweight='bold', color=C_NEG)
        ax_neg.ticklabel_format(axis='x', style='scientific', scilimits=(-3, -3))

    fig.suptitle(f'+ADHD and -ADHD ROIs per Expert — {results["model_type"]} {config_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'signed_rois_per_expert.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'signed_rois_per_expert.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved signed_rois_per_expert.pdf/png")


# ──────────────────────────────────────────────────────────────
# Plot 6: Cross-Model Comparison (all 4 models)
# ──────────────────────────────────────────────────────────────

def plot_cross_model_comparison(out_dir):
    """Compare network saliency across classical/quantum × 2e/4e models."""
    model_dirs = {
        'Classical 4e': 'analysis/expert_saliency_v5_classical_adhd_3',
        'Classical 2e': 'analysis/expert_saliency_v5_classical_adhd_2',
        'Quantum 4e': 'analysis/expert_saliency_v5_quantum_8q_d3_adhd_3',
        'Quantum 2e': 'analysis/expert_saliency_v5_quantum_8q_d3_adhd_2',
    }

    all_results = {}
    for label, dir_path in model_dirs.items():
        p = Path(dir_path)
        if (p / "expert_saliency_results.json").exists():
            with open(p / "expert_saliency_results.json") as f:
                all_results[label] = json.load(f)

    if len(all_results) < 2:
        print("  Skipping cross-model comparison (need at least 2 models)")
        return

    # --- Figure A: Compare 4-expert models (classical vs quantum) ---
    fig_models = [m for m in ['Classical 4e', 'Quantum 4e'] if m in all_results]
    if len(fig_models) == 2:
        config = get_circuit_config('adhd_3')
        expert_names = list(config.keys())

        fig, axes = plt.subplots(1, len(expert_names),
                                  figsize=(5 * len(expert_names), 5))
        if len(expert_names) == 1:
            axes = [axes]

        for exp_idx, exp in enumerate(expert_names):
            ax = axes[exp_idx]
            networks = config[exp]

            x = np.arange(len(networks))
            width = 0.35

            for m_idx, model_label in enumerate(fig_models):
                res = all_results[model_label]
                if exp not in res:
                    continue
                na = res[exp]['network_analysis']

                abs_vals = []
                signed_vals = []
                for net in networks:
                    if net in na:
                        abs_vals.append(na[net]['abs_mean_all'])
                        signed_vals.append(na[net]['signed_mean_all'])
                    else:
                        abs_vals.append(0)
                        signed_vals.append(0)

                # Normalize for comparison (different scale between classical/quantum)
                abs_max = max(abs_vals) if max(abs_vals) > 0 else 1
                norm_vals = [v / abs_max for v in abs_vals]

                offset = -width/2 + m_idx * width
                colors = [C_POS if s > 0 else C_NEG for s in signed_vals]
                bars = ax.bar(x + offset, norm_vals, width,
                             label=model_label if exp_idx == 0 else '',
                             color=colors, alpha=0.7 + m_idx * 0.15,
                             edgecolor='black', linewidth=0.5,
                             hatch='' if m_idx == 0 else '///')

            ax.set_xticks(x)
            ax.set_xticklabels(networks, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Normalized Abs Saliency', fontsize=9)
            ax.set_title(f'{exp}', fontsize=11, fontweight='bold',
                         color=EXPERT_COLORS[exp_idx % len(EXPERT_COLORS)])

        legend_elements = [
            Patch(facecolor='gray', alpha=0.7, label='Classical'),
            Patch(facecolor='gray', alpha=0.85, hatch='///', label='Quantum'),
            Patch(facecolor=C_POS, label='+ADHD'),
            Patch(facecolor=C_NEG, label='-ADHD'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   fontsize=9, bbox_to_anchor=(0.5, -0.06))

        fig.suptitle('Classical vs Quantum 4-Expert — Normalized Network Saliency',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        fig.savefig(out_dir / 'cross_model_4e_comparison.pdf', bbox_inches='tight', dpi=150)
        fig.savefig(out_dir / 'cross_model_4e_comparison.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print("  Saved cross_model_4e_comparison.pdf/png")

    # --- Figure B: Summary table as figure (all models, top ROI per expert) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    table_data = []
    headers = ['Model', 'Expert', 'Top Network (abs)', 'Direction',
               'Top ROI', 'ROI Network', 'ROI Direction', 'p-value']

    for model_label, res in all_results.items():
        config_name = res['circuit_config']
        config = get_circuit_config(config_name)
        for exp in config.keys():
            if exp not in res:
                continue
            na = res[exp]['network_analysis']
            if not na:
                continue
            top_net = max(na.items(), key=lambda x: x[1]['abs_mean_all'])
            top_roi = res[exp]['roi_analysis_top20'][0] if res[exp]['roi_analysis_top20'] else None

            row = [
                model_label, exp,
                top_net[0], top_net[1]['direction'],
                top_roi['region'] if top_roi else '—',
                top_roi['network'] if top_roi else '—',
                top_roi['direction'] if top_roi else '—',
                f"{top_roi['signed_p_value']:.4f}" if top_roi else '—',
            ]
            table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(range(len(headers)))
    table.scale(1, 1.4)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color direction cells
    for i, row in enumerate(table_data, 1):
        for j in [3, 6]:  # direction columns
            if row[j] == '+ADHD':
                table[i, j].set_facecolor('#ffcccc')
            elif row[j] == '-ADHD':
                table[i, j].set_facecolor('#cce5ff')

    fig.suptitle('Cross-Model Summary — Top Network & ROI per Expert',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'cross_model_summary_table.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(out_dir / 'cross_model_summary_table.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved cross_model_summary_table.pdf/png")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize per-expert saliency analysis results')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Path to single model analysis directory')
    parser.add_argument('--cross-model', action='store_true',
                        help='Generate cross-model comparison figures')
    parser.add_argument('--all', action='store_true',
                        help='Run all 4 single-model + cross-model visualizations')
    args = parser.parse_args()

    if args.all:
        dirs = [
            'analysis/expert_saliency_v5_classical_adhd_3',
            'analysis/expert_saliency_v5_classical_adhd_2',
            'analysis/expert_saliency_v5_quantum_8q_d3_adhd_3',
            'analysis/expert_saliency_v5_quantum_8q_d3_adhd_2',
        ]
        for d in dirs:
            if Path(d).exists():
                print(f"\n{'='*60}")
                print(f"Visualizing: {d}")
                print(f"{'='*60}")
                results, base_dir = load_results(d)
                out_dir = base_dir / 'figures'
                out_dir.mkdir(parents=True, exist_ok=True)

                plot_network_saliency_heatmap(results, out_dir)
                plot_expert_top_rois(results, out_dir)
                plot_feature_hierarchy(results, out_dir)
                plot_volcano(results, out_dir)
                plot_signed_rois_per_expert(results, out_dir)

                print(f"  All figures saved to: {out_dir}")

        # Cross-model comparison
        print(f"\n{'='*60}")
        print(f"Cross-model comparison")
        print(f"{'='*60}")
        cross_dir = Path('analysis/expert_saliency_cross_model')
        cross_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_model_comparison(cross_dir)
        print(f"  Saved to: {cross_dir}")

    elif args.cross_model:
        cross_dir = Path('analysis/expert_saliency_cross_model')
        cross_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_model_comparison(cross_dir)
        print(f"\nCross-model figures saved to: {cross_dir}")

    elif args.input_dir:
        results, base_dir = load_results(args.input_dir)
        out_dir = base_dir / 'figures'
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nVisualizing: {args.input_dir}")
        print("=" * 60)

        plot_network_saliency_heatmap(results, out_dir)
        plot_expert_top_rois(results, out_dir)
        plot_feature_hierarchy(results, out_dir)
        plot_volcano(results, out_dir)
        plot_signed_rois_per_expert(results, out_dir)

        print(f"\nAll figures saved to: {out_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
