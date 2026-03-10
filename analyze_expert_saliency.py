#!/usr/bin/env python3
"""
Per-Expert Gradient Saliency Analysis for Circuit MoE Models.

Computes gradient saliency WITHIN each expert independently, answering:
"Within a specific brain circuit, which sub-networks and ROIs does
this expert find most ADHD-relevant?"

This is distinct from the model-level saliency in analyze_circuit_moe.py:
  - Model-level: d(logits)/d(input) — mixes all experts + gating
  - Per-expert:  d(expert_k output)/d(expert_k input) — isolates each expert

Three analysis levels per expert:
  1. Network-level: Which Yeo-17 sub-networks within this circuit are most salient?
  2. ROI-level:     Which individual ROIs does this expert rely on most?
  3. Cross-expert:  How do experts differ in their intra-circuit feature priorities?

Usage:
    python analyze_expert_saliency.py \
        --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49777876.pt \
        --output-dir=analysis/expert_saliency_v5_classical_adhd_3
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.stats import ttest_ind

# scipy.constants must be imported BEFORE pennylane on this system
import scipy.constants  # noqa: F401

from dataloaders.Load_ABCD_fMRI import load_abcd_fmri
from dataloaders.hcp_mmp1_labels import get_roi_name, get_roi_info
from models.yeo17_networks import (
    YEO17_HCP180, get_circuit_config, get_circuit_roi_indices,
)

sys.path.insert(0, "/pscratch/sd/j/junghoon")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_all_seeds(seed=2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transpose_fmri_loaders(loader, batch_size):
    """Transpose (N, T, R) -> (N, R, T) for a single loader."""
    all_x, all_y = [], []
    for batch in loader:
        all_x.append(batch[0])
        all_y.append(batch[1])
    X = torch.cat(all_x, dim=0).permute(0, 2, 1)
    Y = torch.cat(all_y, dim=0)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def build_roi_to_network_map():
    """Return dict mapping ROI index -> Yeo-17 network name."""
    roi_map = {}
    for net_name, indices in YEO17_HCP180.items():
        for idx in indices:
            roi_map[idx] = net_name
    return roi_map


def get_networks_in_circuit(circuit_config_name, circuit_name):
    """Return list of Yeo-17 network names within a circuit."""
    config = get_circuit_config(circuit_config_name)
    return config.get(circuit_name, [])


# ---------------------------------------------------------------------------
# Core: Per-Expert Gradient Saliency
# ---------------------------------------------------------------------------

def compute_per_expert_saliency(model, test_loader, device, circuit_config_name):
    """
    Compute gradient saliency for each expert independently.

    For each expert k:
      1. Run full model forward to get expert_k's output h_k (B, H)
      2. Backprop from h_k.sum() to the FULL input x (B, T, 180)
      3. Extract gradients only at expert k's assigned ROI indices
      4. This gives: "how sensitive is expert k's representation to each of its ROIs?"

    The gradient flows through expert k's own weights only (not through gating
    or other experts), because h_k is computed independently from x[:, :, roi_idx_k].

    Returns:
        dict keyed by circuit_name, each containing:
          - abs_saliency: (N, n_rois_k) absolute gradient magnitude per subject
          - signed_saliency: (N, n_rois_k) signed gradient per subject
          - roi_indices: global ROI indices for this expert
          - labels: (N,) subject labels
    """
    model.eval()

    config = get_circuit_config(circuit_config_name)
    circuits = get_circuit_roi_indices(config)
    n_experts = len(circuits)

    # Storage for per-expert per-subject saliency
    expert_data = {}
    for circuit_name, roi_indices in circuits:
        expert_data[circuit_name] = {
            "abs_batches": [],
            "signed_batches": [],
            "roi_indices": roi_indices,
        }
    all_labels = []

    for batch in tqdm(test_loader, desc="Per-expert saliency", leave=False):
        data, target = batch[0].to(device), batch[1].to(device)
        # data is (B, C, T) from transpose_fmri_loaders
        data_bt_c = data.permute(0, 2, 1)  # (B, T, C=180)

        # For each expert, compute saliency independently
        for expert_idx, (circuit_name, roi_indices) in enumerate(circuits):
            # Create a fresh input tensor that requires grad
            x_input = data_bt_c.detach().clone()
            x_input.requires_grad_(True)

            # Extract this expert's ROI subset
            roi_idx_tensor = getattr(model, f"roi_idx_{expert_idx}")
            x_subset = x_input[:, :, roi_idx_tensor]  # (B, T, n_rois_k)

            # Forward through this expert only
            expert = model.experts[expert_idx]
            if model.model_type == "quantum":
                # Check if v2 (has pool_factor)
                if hasattr(model, 'pool_factor') and model.pool_factor > 1:
                    # v2: temporal pooling before quantum expert
                    # Must match CircuitMoE_v2.py: truncate then pool (no ceil_mode)
                    x_q = x_subset.permute(0, 2, 1)  # (B, n_rois_k, T)
                    T_trunc = (x_q.size(2) // model.pool_factor) * model.pool_factor
                    x_q = x_q[:, :, :T_trunc]
                    x_q = F.avg_pool1d(x_q, kernel_size=model.pool_factor)
                    h_k = expert(x_q)  # (B, H)
                else:
                    h_k = expert(x_subset.permute(0, 2, 1))  # (B, H)
            else:
                h_k = expert(x_subset)  # (B, H)

            # Backprop from expert output to input
            # Using sum() so each sample contributes independently
            h_k.sum().backward()

            # Extract gradients at this expert's ROIs only
            # x_input.grad: (B, T, 180)
            grad_full = x_input.grad  # (B, T, 180)
            grad_expert = grad_full[:, :, roi_idx_tensor]  # (B, T, n_rois_k)

            # Temporal average -> (B, n_rois_k)
            abs_sal = grad_expert.abs().mean(dim=1).detach().cpu().numpy()
            signed_sal = grad_expert.mean(dim=1).detach().cpu().numpy()

            expert_data[circuit_name]["abs_batches"].append(abs_sal)
            expert_data[circuit_name]["signed_batches"].append(signed_sal)

            x_input.requires_grad_(False)

        all_labels.append(target.cpu().numpy())

    labels = np.concatenate(all_labels, axis=0)

    # Concatenate batches
    results = {}
    for circuit_name, roi_indices in circuits:
        ed = expert_data[circuit_name]
        results[circuit_name] = {
            "abs_saliency": np.concatenate(ed["abs_batches"], axis=0),  # (N, n_rois_k)
            "signed_saliency": np.concatenate(ed["signed_batches"], axis=0),
            "roi_indices": ed["roi_indices"],
            "labels": labels,
        }

    return results


# ---------------------------------------------------------------------------
# Analysis: Aggregate per-expert saliency to network and ROI level
# ---------------------------------------------------------------------------

def aggregate_expert_saliency(expert_saliency, circuit_config_name):
    """
    For each expert, aggregate per-subject saliency into:
      1. Network-level summary (mean saliency per Yeo-17 network within circuit)
      2. ROI-level ranking (individual ROIs ranked by saliency)
      3. ADHD+ vs ADHD- comparison at both levels

    Returns:
        dict keyed by circuit_name with detailed analysis
    """
    roi_to_net = build_roi_to_network_map()
    config = get_circuit_config(circuit_config_name)

    all_results = {}

    for circuit_name, data in expert_saliency.items():
        abs_sal = data["abs_saliency"]      # (N, n_rois_k)
        signed_sal = data["signed_saliency"]  # (N, n_rois_k)
        roi_indices = data["roi_indices"]
        labels = data["labels"]

        pos_mask = labels == 1  # ADHD+
        neg_mask = labels == 0  # ADHD-

        # --- Network-level within this expert ---
        networks_in_circuit = config.get(circuit_name, [])
        network_analysis = {}

        for net_name in networks_in_circuit:
            # Find which LOCAL indices in this expert map to this network
            net_global_rois = set(YEO17_HCP180[net_name])
            local_indices = [
                local_idx for local_idx, global_idx in enumerate(roi_indices)
                if global_idx in net_global_rois
            ]

            if not local_indices:
                continue

            # Absolute saliency
            net_abs_all = abs_sal[:, local_indices].mean(axis=1)  # (N,)
            net_abs_pos = abs_sal[pos_mask][:, local_indices].mean(axis=1)
            net_abs_neg = abs_sal[neg_mask][:, local_indices].mean(axis=1)

            # Signed saliency
            net_signed_all = signed_sal[:, local_indices].mean(axis=1)
            net_signed_pos = signed_sal[pos_mask][:, local_indices].mean(axis=1)
            net_signed_neg = signed_sal[neg_mask][:, local_indices].mean(axis=1)

            # t-test: does this network's saliency differ between ADHD+ and ADHD-?
            t_abs, p_abs = ttest_ind(net_abs_pos, net_abs_neg, equal_var=False)
            t_signed, p_signed = ttest_ind(net_signed_pos, net_signed_neg,
                                           equal_var=False)

            direction = "+ADHD" if float(np.mean(net_signed_all)) > 0 else "-ADHD"

            network_analysis[net_name] = {
                "n_rois": len(local_indices),
                "abs_mean_all": float(np.mean(net_abs_all)),
                "abs_mean_pos": float(np.mean(net_abs_pos)),
                "abs_mean_neg": float(np.mean(net_abs_neg)),
                "abs_t_stat": float(t_abs),
                "abs_p_value": float(p_abs),
                "signed_mean_all": float(np.mean(net_signed_all)),
                "signed_mean_pos": float(np.mean(net_signed_pos)),
                "signed_mean_neg": float(np.mean(net_signed_neg)),
                "signed_t_stat": float(t_signed),
                "signed_p_value": float(p_signed),
                "direction": direction,
            }

        # --- ROI-level within this expert ---
        roi_analysis = []
        for local_idx, global_idx in enumerate(roi_indices):
            roi_abs_all = float(abs_sal[:, local_idx].mean())
            roi_abs_pos = float(abs_sal[pos_mask, local_idx].mean())
            roi_abs_neg = float(abs_sal[neg_mask, local_idx].mean())

            roi_signed_all = float(signed_sal[:, local_idx].mean())
            roi_signed_pos = float(signed_sal[pos_mask, local_idx].mean())
            roi_signed_neg = float(signed_sal[neg_mask, local_idx].mean())

            # Per-ROI t-test
            t_stat, p_val = ttest_ind(
                abs_sal[pos_mask, local_idx],
                abs_sal[neg_mask, local_idx],
                equal_var=False,
            )
            t_signed_stat, p_signed_val = ttest_ind(
                signed_sal[pos_mask, local_idx],
                signed_sal[neg_mask, local_idx],
                equal_var=False,
            )

            roi_info = get_roi_info(global_idx)
            direction = "+ADHD" if roi_signed_all > 0 else "-ADHD"

            roi_analysis.append({
                "global_roi_idx": int(global_idx),
                "local_roi_idx": int(local_idx),
                "region": roi_info["name"],
                "long_name": roi_info["long_name"],
                "lobe": roi_info["lobe"],
                "network": roi_to_net.get(global_idx, "unknown"),
                "abs_mean_all": roi_abs_all,
                "abs_mean_pos": roi_abs_pos,
                "abs_mean_neg": roi_abs_neg,
                "abs_t_stat": float(t_stat),
                "abs_p_value": float(p_val),
                "signed_mean_all": roi_signed_all,
                "signed_mean_pos": roi_signed_pos,
                "signed_mean_neg": roi_signed_neg,
                "signed_t_stat": float(t_signed_stat),
                "signed_p_value": float(p_signed_val),
                "direction": direction,
            })

        # Sort by absolute saliency (descending)
        roi_analysis.sort(key=lambda x: x["abs_mean_all"], reverse=True)

        all_results[circuit_name] = {
            "network_analysis": network_analysis,
            "roi_analysis": roi_analysis,
            "n_rois": len(roi_indices),
            "n_adhd_pos": int(pos_mask.sum()),
            "n_adhd_neg": int(neg_mask.sum()),
        }

    return all_results


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(aggregated, circuit_config_name, model_type, output_dir):
    """Generate a markdown report for per-expert saliency analysis."""

    config = get_circuit_config(circuit_config_name)
    circuits = get_circuit_roi_indices(config)
    circuit_names = [name for name, _ in circuits]

    lines = []
    lines.append("# Per-Expert Gradient Saliency Analysis")
    lines.append("")
    lines.append(f"**Model**: {model_type} | **Config**: {circuit_config_name} | "
                 f"**Experts**: {len(circuits)}")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("For each expert, we compute ∂(expert_k output)/∂(expert_k input ROIs), "
                 "isolating what each expert has learned to attend to within its assigned "
                 "brain circuit. This is distinct from model-level saliency "
                 "(∂logits/∂input), which mixes contributions from all experts and gating.")
    lines.append("")
    lines.append("- **Absolute saliency**: Magnitude of gradient (how much this ROI "
                 "matters to this expert, regardless of direction)")
    lines.append("- **Signed saliency**: Direction of gradient (+ADHD = increasing "
                 "activity pushes expert output in ADHD-associated direction)")
    lines.append("- **Statistical tests**: Welch's t-test comparing ADHD+ vs ADHD- "
                 "subjects at network and ROI levels")
    lines.append("")

    # Per-expert sections
    for circuit_name in circuit_names:
        if circuit_name not in aggregated:
            continue

        data = aggregated[circuit_name]
        net_analysis = data["network_analysis"]
        roi_analysis = data["roi_analysis"]
        networks_in_circuit = config.get(circuit_name, [])

        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Expert: {circuit_name} ({data['n_rois']} ROIs)")
        lines.append(f"")
        lines.append(f"Networks: {', '.join(networks_in_circuit)}")
        lines.append(f"")

        # Network-level table
        lines.append(f"### Intra-Circuit Network Saliency")
        lines.append(f"")
        lines.append(f"Which Yeo-17 sub-networks does the {circuit_name} expert "
                     f"rely on most for its representation?")
        lines.append(f"")
        lines.append("| Network | n_ROIs | Abs Saliency | Signed | Direction | "
                     "ADHD+ vs ADHD- (signed p) |")
        lines.append("|---------|:------:|:------------:|:------:|:---------:|"
                     ":------------------------:|")

        sorted_nets = sorted(net_analysis.items(),
                           key=lambda x: x[1]["abs_mean_all"], reverse=True)
        for net_name, ns in sorted_nets:
            sig = "***" if ns["signed_p_value"] < 0.001 else \
                  "**" if ns["signed_p_value"] < 0.01 else \
                  "*" if ns["signed_p_value"] < 0.05 else "n.s."
            lines.append(
                f"| **{net_name}** | {ns['n_rois']} | "
                f"{ns['abs_mean_all']:.6f} | {ns['signed_mean_all']:+.6f} | "
                f"{ns['direction']} | p={ns['signed_p_value']:.4f} {sig} |"
            )

        lines.append(f"")

        # Top 10 ROIs by absolute saliency
        lines.append(f"### Top 10 ROIs by Absolute Saliency")
        lines.append(f"")
        lines.append("| Rank | ROI | Region | Long Name | Lobe | Network | "
                     "Abs Saliency | Signed | Direction | Signed p |")
        lines.append("|:----:|:---:|--------|-----------|------|---------|"
                     ":----------:|:------:|:---------:|:--------:|")

        for rank, roi in enumerate(roi_analysis[:10], 1):
            sig = "*" if roi["signed_p_value"] < 0.05 else ""
            lines.append(
                f"| {rank} | {roi['global_roi_idx']} | {roi['region']} | "
                f"{roi['long_name']} | {roi['lobe']} | {roi['network']} | "
                f"{roi['abs_mean_all']:.6f} | {roi['signed_mean_all']:+.6f} | "
                f"{roi['direction']} | {roi['signed_p_value']:.4f}{sig} |"
            )

        lines.append(f"")

        # Top 5 +ADHD and Top 5 -ADHD ROIs (by signed saliency)
        sorted_positive = sorted(roi_analysis,
                                key=lambda x: x["signed_mean_all"], reverse=True)
        sorted_negative = sorted(roi_analysis,
                                key=lambda x: x["signed_mean_all"])

        lines.append(f"### Top 5 +ADHD ROIs (strongest positive association)")
        lines.append(f"")
        lines.append("| ROI | Region | Network | Signed Saliency | p-value |")
        lines.append("|:---:|--------|---------|:---------------:|:-------:|")
        for roi in sorted_positive[:5]:
            sig = "*" if roi["signed_p_value"] < 0.05 else ""
            lines.append(
                f"| {roi['global_roi_idx']} | {roi['region']} ({roi['long_name']}) | "
                f"{roi['network']} | {roi['signed_mean_all']:+.6f} | "
                f"{roi['signed_p_value']:.4f}{sig} |"
            )
        lines.append(f"")

        lines.append(f"### Top 5 -ADHD ROIs (strongest negative association)")
        lines.append(f"")
        lines.append("| ROI | Region | Network | Signed Saliency | p-value |")
        lines.append("|:---:|--------|---------|:---------------:|:-------:|")
        for roi in sorted_negative[:5]:
            sig = "*" if roi["signed_p_value"] < 0.05 else ""
            lines.append(
                f"| {roi['global_roi_idx']} | {roi['region']} ({roi['long_name']}) | "
                f"{roi['network']} | {roi['signed_mean_all']:+.6f} | "
                f"{roi['signed_p_value']:.4f}{sig} |"
            )
        lines.append(f"")

    # --- Cross-Expert Comparison ---
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Cross-Expert Comparison")
    lines.append(f"")
    lines.append(f"### Dominant Network Per Expert")
    lines.append(f"")
    lines.append("Which sub-network does each expert prioritize most?")
    lines.append("")
    lines.append("| Expert | Top Network (abs) | Abs Saliency | "
                 "Top Network (signed) | Signed | Direction |")
    lines.append("|--------|:-----------------:|:------------:|"
                 ":--------------------:|:------:|:---------:|")

    for circuit_name in circuit_names:
        if circuit_name not in aggregated:
            continue
        net_analysis = aggregated[circuit_name]["network_analysis"]
        if not net_analysis:
            continue

        top_abs = max(net_analysis.items(), key=lambda x: x[1]["abs_mean_all"])
        top_signed = max(net_analysis.items(),
                        key=lambda x: abs(x[1]["signed_mean_all"]))

        lines.append(
            f"| {circuit_name} | {top_abs[0]} | {top_abs[1]['abs_mean_all']:.6f} | "
            f"{top_signed[0]} | {top_signed[1]['signed_mean_all']:+.6f} | "
            f"{top_signed[1]['direction']} |"
        )

    lines.append(f"")

    # Network saliency ranked within each expert (compact cross-expert view)
    lines.append(f"### Intra-Circuit Feature Hierarchy (All Experts)")
    lines.append(f"")
    lines.append("Networks ranked by absolute saliency within each expert. "
                 "This reveals whether experts prioritize their 'expected' "
                 "sub-networks or learn unexpected feature hierarchies.")
    lines.append("")

    for circuit_name in circuit_names:
        if circuit_name not in aggregated:
            continue
        net_analysis = aggregated[circuit_name]["network_analysis"]
        sorted_nets = sorted(net_analysis.items(),
                           key=lambda x: x[1]["abs_mean_all"], reverse=True)
        ranking_str = " > ".join(
            f"{name} ({ns['abs_mean_all']:.4f}, {ns['direction']})"
            for name, ns in sorted_nets
        )
        lines.append(f"- **{circuit_name}**: {ranking_str}")

    lines.append(f"")

    # --- Significant Findings Summary ---
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Significant Findings (p < 0.05)")
    lines.append(f"")

    sig_count = 0
    for circuit_name in circuit_names:
        if circuit_name not in aggregated:
            continue

        # Significant networks
        net_analysis = aggregated[circuit_name]["network_analysis"]
        for net_name, ns in net_analysis.items():
            if ns["signed_p_value"] < 0.05:
                sig_count += 1
                lines.append(
                    f"- **{circuit_name} expert → {net_name}**: "
                    f"Signed saliency differs between ADHD+ and ADHD- "
                    f"(p={ns['signed_p_value']:.4f}, direction={ns['direction']})"
                )

        # Significant ROIs (top by significance)
        roi_analysis = aggregated[circuit_name]["roi_analysis"]
        sig_rois = [r for r in roi_analysis if r["signed_p_value"] < 0.05]
        sig_rois.sort(key=lambda x: x["signed_p_value"])
        for roi in sig_rois[:3]:
            sig_count += 1
            lines.append(
                f"- **{circuit_name} expert → ROI {roi['region']}** "
                f"({roi['network']}): p={roi['signed_p_value']:.4f}, "
                f"{roi['direction']}, signed={roi['signed_mean_all']:+.6f}"
            )

    if sig_count == 0:
        lines.append("No network-level or ROI-level findings reached p < 0.05.")
        lines.append("This is consistent with the load-balancing loss suppressing "
                     "class-conditional expert specialization.")

    lines.append(f"")

    report_path = os.path.join(output_dir, "expert_saliency_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Per-expert gradient saliency analysis for Circuit MoE",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="analysis/expert_saliency",
                        help="Directory for analysis outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load checkpoint and reconstruct model ---
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    model_type = ckpt_args["model_type"]
    circuit_config = ckpt_args["circuit_config"]
    seed = ckpt_args.get("seed", 2025)
    set_all_seeds(seed)

    print(f"Model: {model_type}, Circuit: {circuit_config}, Seed: {seed}")

    # Determine if this is a v2 model (has pool_factor)
    is_v2 = "pool_factor" in ckpt_args and ckpt_args["pool_factor"] is not None

    # --- Load data ---
    print("\nLoading ABCD fMRI data...")
    sample_sz = ckpt_args.get("sample_size", 0)
    sample_sz = sample_sz if sample_sz and sample_sz > 0 else None
    _, _, test_loader, input_dim = load_abcd_fmri(
        seed=seed, device=device, batch_size=args.batch_size,
        parcel_type=ckpt_args.get("parcel_type", "HCP180"),
        target_phenotype=ckpt_args.get("target_phenotype", "ADHD_label"),
        task_type=ckpt_args.get("task_type", "binary"),
        sample_size=sample_sz,
    )
    test_loader = transpose_fmri_loaders(test_loader, args.batch_size)
    n_samples, n_timesteps, n_channels = input_dim
    print(f"Test set: {n_channels} channels, {n_timesteps} timesteps")

    # --- Reconstruct model ---
    if is_v2:
        from models.CircuitMoE_v2 import CircuitMoE
        pool_factor = ckpt_args.get("pool_factor", 10)
    else:
        from models.CircuitMoE import CircuitMoE
        pool_factor = 1

    model_kwargs = dict(
        circuit_config_name=circuit_config,
        time_points=n_timesteps,
        expert_hidden_dim=ckpt_args.get("expert_hidden_dim", 64),
        model_type=model_type,
        expert_layers=ckpt_args.get("expert_layers", 2),
        nhead=ckpt_args.get("nhead", 4),
        n_qubits=ckpt_args.get("n_qubits", 8),
        n_ansatz_layers=ckpt_args.get("n_ansatz_layers", 2),
        degree=ckpt_args.get("degree", 3),
        gating_noise_std=0.0,
        total_channels=n_channels,
        num_classes=ckpt_args.get("num_classes", 2),
        dropout=0.0,
        device=device,
    )
    if is_v2:
        model_kwargs["pool_factor"] = pool_factor
        model_kwargs["grad_scale"] = False

    model = CircuitMoE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded. Experts: {model.num_experts} ({', '.join(model.circuit_names)})")

    # --- Compute per-expert saliency ---
    print("\n" + "=" * 70)
    print("Computing per-expert gradient saliency...")
    print("=" * 70)

    expert_saliency = compute_per_expert_saliency(
        model, test_loader, device, circuit_config,
    )

    # --- Aggregate to network and ROI level ---
    print("\nAggregating to network and ROI levels...")
    aggregated = aggregate_expert_saliency(expert_saliency, circuit_config)

    # --- Print summary ---
    config = get_circuit_config(circuit_config)
    for circuit_name, data in aggregated.items():
        print(f"\n  Expert: {circuit_name} ({data['n_rois']} ROIs)")

        # Network ranking
        net_analysis = data["network_analysis"]
        sorted_nets = sorted(net_analysis.items(),
                           key=lambda x: x[1]["abs_mean_all"], reverse=True)
        print(f"    Network ranking (abs saliency):")
        for net_name, ns in sorted_nets:
            sig = "*" if ns["signed_p_value"] < 0.05 else " "
            print(f"      {sig} {net_name:20s}: abs={ns['abs_mean_all']:.6f}, "
                  f"signed={ns['signed_mean_all']:+.6f} ({ns['direction']}), "
                  f"p={ns['signed_p_value']:.4f}")

        # Top 3 ROIs
        print(f"    Top 3 ROIs (abs saliency):")
        for roi in data["roi_analysis"][:3]:
            sig = "*" if roi["signed_p_value"] < 0.05 else " "
            print(f"      {sig} {roi['region']:15s} ({roi['network']:20s}): "
                  f"abs={roi['abs_mean_all']:.6f}, "
                  f"signed={roi['signed_mean_all']:+.6f} ({roi['direction']})")

    # --- Generate report ---
    print("\n" + "=" * 70)
    print("Generating report...")
    print("=" * 70)
    generate_report(aggregated, circuit_config, model_type, args.output_dir)

    # --- Save raw results as JSON ---
    json_results = {
        "model_type": model_type,
        "circuit_config": circuit_config,
        "checkpoint": args.checkpoint,
        "seed": seed,
        "analysis_type": "per_expert_saliency",
    }
    for circuit_name, data in aggregated.items():
        json_results[circuit_name] = {
            "n_rois": data["n_rois"],
            "n_adhd_pos": data["n_adhd_pos"],
            "n_adhd_neg": data["n_adhd_neg"],
            "network_analysis": data["network_analysis"],
            "roi_analysis_top20": data["roi_analysis"][:20],
            "roi_signed_top10_positive": sorted(
                data["roi_analysis"],
                key=lambda x: x["signed_mean_all"], reverse=True
            )[:10],
            "roi_signed_top10_negative": sorted(
                data["roi_analysis"],
                key=lambda x: x["signed_mean_all"]
            )[:10],
        }

    json_path = os.path.join(args.output_dir, "expert_saliency_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Raw results saved to: {json_path}")

    # --- Save per-subject saliency for downstream analysis ---
    npz_data = {"labels": expert_saliency[list(expert_saliency.keys())[0]]["labels"]}
    for circuit_name, data in expert_saliency.items():
        safe_name = circuit_name.replace(" ", "_")
        npz_data[f"{safe_name}_abs_saliency"] = data["abs_saliency"]
        npz_data[f"{safe_name}_signed_saliency"] = data["signed_saliency"]
        npz_data[f"{safe_name}_roi_indices"] = np.array(data["roi_indices"])

    npz_path = os.path.join(args.output_dir, "expert_saliency_per_subject.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Per-subject data saved to: {npz_path}")


if __name__ == "__main__":
    main()
