#!/usr/bin/env python3
"""
Post-hoc interpretability analysis for Circuit MoE models.

Produces a 3-level interpretability hierarchy:
  Circuit level  — gate weight differences between ADHD+ and ADHD- subjects
  Network level  — gradient saliency grouped by Yeo-17 network within each circuit
  ROI level      — per-ROI gradient saliency and input projection weight magnitudes

Usage:
    python analyze_circuit_moe.py \
        --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49731003.pt \
        --output-dir=analysis/classical_adhd_3

    python analyze_circuit_moe.py \
        --checkpoint=checkpoints/CircuitMoE_quantum_adhd_3_49767122.pt \
        --output-dir=analysis/quantum_v2_adhd_3
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

# scipy.constants must be imported BEFORE pennylane on this system
import scipy.constants  # noqa: F401

from dataloaders.Load_ABCD_fMRI import load_abcd_fmri
from dataloaders.hcp_mmp1_labels import get_roi_name
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
    """Return dict mapping ROI index -> network name."""
    roi_map = {}
    for net_name, indices in YEO17_HCP180.items():
        for idx in indices:
            roi_map[idx] = net_name
    return roi_map


def build_roi_to_circuit_map(circuit_config_name):
    """Return dict mapping ROI index -> circuit name."""
    config = get_circuit_config(circuit_config_name)
    circuits = get_circuit_roi_indices(config)
    roi_map = {}
    for circuit_name, indices in circuits:
        for idx in indices:
            roi_map[idx] = circuit_name
    return roi_map


# ---------------------------------------------------------------------------
# Analysis 1: Gate Weights by Class
# ---------------------------------------------------------------------------

def analyze_gate_weights(model, test_loader, device, circuit_names):
    """
    Compute per-sample gate weights and split by ADHD+ / ADHD-.

    Returns:
        dict with keys: gate_weights_pos, gate_weights_neg, stats
    """
    model.eval()
    all_gates = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Gate weights", leave=False):
            data, target = batch[0].to(device), batch[1].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            _, gate_weights = model(data)
            all_gates.append(gate_weights.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    gates = np.concatenate(all_gates, axis=0)  # (N, K)
    labels = np.concatenate(all_labels, axis=0)  # (N,)

    pos_mask = labels == 1
    neg_mask = labels == 0

    gates_pos = gates[pos_mask]  # ADHD+
    gates_neg = gates[neg_mask]  # ADHD-

    # Per-expert statistics
    stats = {}
    for i, name in enumerate(circuit_names):
        pos_mean = float(gates_pos[:, i].mean())
        neg_mean = float(gates_neg[:, i].mean())
        pos_std = float(gates_pos[:, i].std())
        neg_std = float(gates_neg[:, i].std())

        # Welch's t-test
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(gates_pos[:, i], gates_neg[:, i],
                                   equal_var=False)

        stats[name] = {
            "adhd_pos_mean": pos_mean,
            "adhd_neg_mean": neg_mean,
            "adhd_pos_std": pos_std,
            "adhd_neg_std": neg_std,
            "diff": pos_mean - neg_mean,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        }

    return {
        "gate_weights_pos": gates_pos,
        "gate_weights_neg": gates_neg,
        "labels": labels,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Gradient Saliency (ROI + Network level)
# ---------------------------------------------------------------------------

def analyze_gradient_saliency(model, test_loader, device, circuit_config_name):
    """
    Compute gradient-based saliency maps: dOutput/dInput for each sample.
    For binary classification, we use the gradient of the predicted logit
    w.r.t. the raw input (B, T, 180).

    Returns:
        dict with per-ROI saliency for ADHD+ and ADHD- subjects
    """
    model.eval()
    all_saliency = []
    all_signed_saliency = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Gradient saliency", leave=False):
        data, target = batch[0].to(device), batch[1].to(device)
        data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        data.requires_grad_(True)

        logits, _ = model(data)

        # For binary: gradient of the sigmoid logit for each sample
        # We use sum so each sample contributes independently
        logits.sum().backward()

        # Absolute saliency: magnitude of influence (how much this ROI matters)
        # data.grad: (B, T, 180) -> mean over T -> (B, 180)
        saliency = data.grad.abs().mean(dim=1).detach().cpu().numpy()
        all_saliency.append(saliency)

        # Signed saliency: direction of influence (+/- relationship with ADHD)
        # Positive = increasing this ROI's signal pushes prediction toward ADHD+
        # Negative = increasing this ROI's signal pushes prediction toward ADHD-
        signed = data.grad.mean(dim=1).detach().cpu().numpy()
        all_signed_saliency.append(signed)

        all_labels.append(target.cpu().numpy())

        data.requires_grad_(False)

    saliency = np.concatenate(all_saliency, axis=0)  # (N, 180)
    signed_saliency = np.concatenate(all_signed_saliency, axis=0)  # (N, 180)
    labels = np.concatenate(all_labels, axis=0)

    pos_mask = labels == 1
    neg_mask = labels == 0

    saliency_pos = saliency[pos_mask].mean(axis=0)  # (180,)
    saliency_neg = saliency[neg_mask].mean(axis=0)  # (180,)
    saliency_all = saliency.mean(axis=0)

    signed_pos = signed_saliency[pos_mask].mean(axis=0)  # (180,)
    signed_neg = signed_saliency[neg_mask].mean(axis=0)  # (180,)
    signed_all = signed_saliency.mean(axis=0)

    # Group by network
    roi_to_net = build_roi_to_network_map()
    network_saliency = {}
    for net_name in YEO17_HCP180:
        indices = YEO17_HCP180[net_name]
        network_saliency[net_name] = {
            "all_mean": float(saliency_all[indices].mean()),
            "adhd_pos_mean": float(saliency_pos[indices].mean()),
            "adhd_neg_mean": float(saliency_neg[indices].mean()),
            "diff": float(saliency_pos[indices].mean() - saliency_neg[indices].mean()),
            "signed_all": float(signed_all[indices].mean()),
            "signed_pos": float(signed_pos[indices].mean()),
            "signed_neg": float(signed_neg[indices].mean()),
            "n_rois": len(indices),
        }

    # Group by circuit
    roi_to_circuit = build_roi_to_circuit_map(circuit_config_name)
    config = get_circuit_config(circuit_config_name)
    circuits = get_circuit_roi_indices(config)
    circuit_saliency = {}
    for circuit_name, indices in circuits:
        circuit_saliency[circuit_name] = {
            "all_mean": float(saliency_all[indices].mean()),
            "adhd_pos_mean": float(saliency_pos[indices].mean()),
            "adhd_neg_mean": float(saliency_neg[indices].mean()),
            "diff": float(saliency_pos[indices].mean() - saliency_neg[indices].mean()),
            "signed_all": float(signed_all[indices].mean()),
            "signed_pos": float(signed_pos[indices].mean()),
            "signed_neg": float(signed_neg[indices].mean()),
            "n_rois": len(indices),
        }

    # Per-ROI saliency (top 20)
    roi_saliency = []
    for roi_idx in range(180):
        roi_saliency.append({
            "roi_idx": int(roi_idx),
            "region": get_roi_name(roi_idx),
            "network": roi_to_net.get(roi_idx, "unknown"),
            "circuit": roi_to_circuit.get(roi_idx, "unknown"),
            "saliency_all": float(saliency_all[roi_idx]),
            "saliency_pos": float(saliency_pos[roi_idx]),
            "saliency_neg": float(saliency_neg[roi_idx]),
            "diff": float(saliency_pos[roi_idx] - saliency_neg[roi_idx]),
            "signed_all": float(signed_all[roi_idx]),
            "signed_pos": float(signed_pos[roi_idx]),
            "signed_neg": float(signed_neg[roi_idx]),
        })
    roi_saliency.sort(key=lambda x: x["saliency_all"], reverse=True)

    return {
        "circuit_saliency": circuit_saliency,
        "network_saliency": network_saliency,
        "roi_saliency": roi_saliency,
        "saliency_pos_raw": saliency_pos,
        "saliency_neg_raw": saliency_neg,
        "signed_pos_raw": signed_pos,
        "signed_neg_raw": signed_neg,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Input Projection Weights
# ---------------------------------------------------------------------------

def analyze_input_weights(model, circuit_config_name):
    """
    Extract input projection weight magnitudes per ROI within each expert.
    Groups by Yeo-17 network to show which sub-networks each expert relies on.

    Works for both classical (CircuitExpert.input_projection) and quantum
    (QuantumTSTransformer.feature_projection) experts.
    """
    config = get_circuit_config(circuit_config_name)
    circuits = get_circuit_roi_indices(config)
    roi_to_net = build_roi_to_network_map()

    expert_weight_analysis = {}

    for i, (circuit_name, roi_indices) in enumerate(circuits):
        expert = model.experts[i]

        # Get the first linear layer weights
        if hasattr(expert, 'input_projection'):
            W = expert.input_projection.weight.detach().cpu().numpy()  # (H, n_rois)
        elif hasattr(expert, 'feature_projection'):
            W = expert.feature_projection.weight.detach().cpu().numpy()  # (n_rots, n_rois)
        else:
            print(f"  Warning: Expert {i} ({circuit_name}) has no recognized projection layer")
            continue

        # Per-ROI weight magnitude: L2 norm of column
        roi_weight_norms = np.linalg.norm(W, axis=0)  # (n_rois,)

        # Map local ROI index to global ROI index and network
        roi_weights = []
        for local_idx, global_idx in enumerate(roi_indices):
            roi_weights.append({
                "global_roi_idx": int(global_idx),
                "region": get_roi_name(global_idx),
                "local_roi_idx": int(local_idx),
                "network": roi_to_net.get(global_idx, "unknown"),
                "weight_norm": float(roi_weight_norms[local_idx]),
            })

        # Group by network within this expert's circuit
        network_groups = {}
        for rw in roi_weights:
            net = rw["network"]
            if net not in network_groups:
                network_groups[net] = []
            network_groups[net].append(rw["weight_norm"])

        network_summary = {}
        for net, norms in network_groups.items():
            network_summary[net] = {
                "mean_weight_norm": float(np.mean(norms)),
                "std_weight_norm": float(np.std(norms)),
                "max_weight_norm": float(np.max(norms)),
                "n_rois": len(norms),
            }

        # Sort ROIs by weight norm (descending)
        roi_weights.sort(key=lambda x: x["weight_norm"], reverse=True)

        expert_weight_analysis[circuit_name] = {
            "network_summary": network_summary,
            "top_rois": roi_weights[:10],
            "all_rois": roi_weights,
        }

    return expert_weight_analysis


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(gate_results, saliency_results, weight_results,
                    circuit_config_name, model_type, output_dir):
    """Generate a markdown report summarizing all analyses."""

    config = get_circuit_config(circuit_config_name)
    circuits = get_circuit_roi_indices(config)
    circuit_names = [name for name, _ in circuits]

    lines = []
    lines.append(f"# Circuit MoE Interpretability Analysis")
    lines.append(f"")
    lines.append(f"**Model**: {model_type} | **Config**: {circuit_config_name} | "
                 f"**Experts**: {len(circuits)}")
    lines.append(f"")

    # --- Circuit Level: Gate Weights ---
    lines.append(f"## 1. Circuit-Level Analysis: Gate Weights by Class")
    lines.append(f"")
    lines.append(f"Gate weights indicate how much each circuit expert contributes to "
                 f"the final prediction. Differences between ADHD+ and ADHD- subjects "
                 f"reveal which circuits are differentially engaged.")
    lines.append(f"")
    lines.append(f"| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |")
    lines.append(f"|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|")

    for name in circuit_names:
        s = gate_results["stats"][name]
        sig = "**" if s["p_value"] < 0.05 else ""
        interp = ""
        if s["p_value"] < 0.05:
            direction = "higher" if s["diff"] > 0 else "lower"
            interp = f"ADHD+ routes {direction} to {name}"
        else:
            interp = "No significant difference"
        lines.append(f"| {sig}{name}{sig} | {s['adhd_pos_mean']:.4f} | "
                     f"{s['adhd_neg_mean']:.4f} | {s['diff']:+.4f} | "
                     f"{s['t_stat']:.3f} | {s['p_value']:.4f} | {interp} |")

    lines.append(f"")

    # --- Circuit Level: Gradient Saliency (Absolute) ---
    lines.append(f"## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)")
    lines.append(f"")
    lines.append(f"Absolute gradient saliency measures how much each ROI's input signal "
                 f"influences the model output (magnitude only). Higher = more influential.")
    lines.append(f"")
    lines.append(f"| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |")
    lines.append(f"|---------|:--------------:|:-----:|:-----:|:----:|:------:|")

    for name, s in saliency_results["circuit_saliency"].items():
        lines.append(f"| {name} | {s['all_mean']:.6f} | {s['adhd_pos_mean']:.6f} | "
                     f"{s['adhd_neg_mean']:.6f} | {s['diff']:+.6f} | {s['n_rois']} |")

    lines.append(f"")

    # --- Circuit Level: Signed Gradient Saliency ---
    lines.append(f"## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)")
    lines.append(f"")
    lines.append(f"Signed gradient saliency preserves the direction of influence. "
                 f"**Positive** = increasing this circuit's signal pushes the prediction "
                 f"toward ADHD+. **Negative** = pushes toward ADHD-.")
    lines.append(f"")
    lines.append(f"| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |")
    lines.append(f"|---------|:------------:|:-----:|:-----:|:---------:|:------:|")

    for name, s in saliency_results["circuit_saliency"].items():
        direction = "positive (+ADHD)" if s["signed_all"] > 0 else "negative (-ADHD)"
        lines.append(f"| {name} | {s['signed_all']:+.6f} | {s['signed_pos']:+.6f} | "
                     f"{s['signed_neg']:+.6f} | {direction} | {s['n_rois']} |")

    lines.append(f"")

    # --- Network Level: Absolute ---
    lines.append(f"## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network")
    lines.append(f"")
    lines.append(f"Absolute saliency aggregated by Yeo-17 network (magnitude of influence).")
    lines.append(f"")
    lines.append(f"| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |")
    lines.append(f"|---------|:--------------:|:-----:|:-----:|:----:|:------:|")

    sorted_nets = sorted(saliency_results["network_saliency"].items(),
                        key=lambda x: x[1]["all_mean"], reverse=True)
    for net_name, s in sorted_nets:
        lines.append(f"| {net_name} | {s['all_mean']:.6f} | {s['adhd_pos_mean']:.6f} | "
                     f"{s['adhd_neg_mean']:.6f} | {s['diff']:+.6f} | {s['n_rois']} |")

    lines.append(f"")

    # --- Network Level: Signed ---
    lines.append(f"## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)")
    lines.append(f"")
    lines.append(f"Signed saliency by Yeo-17 network. Positive = network activity positively "
                 f"associated with ADHD prediction. Negative = negatively associated.")
    lines.append(f"")
    lines.append(f"| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |")
    lines.append(f"|---------|:------------:|:-----:|:-----:|:---------:|:------:|")

    sorted_nets_signed = sorted(saliency_results["network_saliency"].items(),
                                key=lambda x: x[1]["signed_all"], reverse=True)
    for net_name, s in sorted_nets_signed:
        direction = "+ADHD" if s["signed_all"] > 0 else "-ADHD"
        lines.append(f"| {net_name} | {s['signed_all']:+.6f} | {s['signed_pos']:+.6f} | "
                     f"{s['signed_neg']:+.6f} | {direction} | {s['n_rois']} |")

    lines.append(f"")

    # --- Network Level: Input Projection Weights ---
    lines.append(f"## 4. Network-Level Analysis: Input Projection Weights per Expert")
    lines.append(f"")
    lines.append(f"Weight magnitudes of the first linear layer in each expert, "
                 f"grouped by Yeo-17 network. Shows which sub-networks each expert "
                 f"learns to attend to.")
    lines.append(f"")

    for circuit_name in circuit_names:
        if circuit_name not in weight_results:
            continue
        wr = weight_results[circuit_name]
        lines.append(f"### Expert: {circuit_name}")
        lines.append(f"")
        lines.append(f"| Network | Mean Weight | Std | Max | n_ROIs |")
        lines.append(f"|---------|:----------:|:---:|:---:|:------:|")

        sorted_nets = sorted(wr["network_summary"].items(),
                            key=lambda x: x[1]["mean_weight_norm"], reverse=True)
        for net_name, ns in sorted_nets:
            lines.append(f"| {net_name} | {ns['mean_weight_norm']:.4f} | "
                         f"{ns['std_weight_norm']:.4f} | {ns['max_weight_norm']:.4f} | "
                         f"{ns['n_rois']} |")

        lines.append(f"")
        lines.append(f"**Top 5 ROIs by weight magnitude:**")
        lines.append(f"")
        lines.append(f"| ROI | Region | Network | Weight Norm |")
        lines.append(f"|:---:|--------|---------|:----------:|")
        for rw in wr["top_rois"][:5]:
            lines.append(f"| {rw['global_roi_idx']} | {rw['region']} | {rw['network']} | "
                         f"{rw['weight_norm']:.4f} |")
        lines.append(f"")

    # --- ROI Level: Absolute ---
    lines.append(f"## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency")
    lines.append(f"")
    lines.append(f"| Rank | ROI | Region | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |")
    lines.append(f"|:----:|:---:|--------|---------|---------|:--------:|:-----:|:-----:|:----:|")

    for rank, roi in enumerate(saliency_results["roi_saliency"][:20], 1):
        lines.append(f"| {rank} | {roi['roi_idx']} | {roi['region']} | {roi['network']} | "
                     f"{roi['circuit']} | {roi['saliency_all']:.6f} | "
                     f"{roi['saliency_pos']:.6f} | {roi['saliency_neg']:.6f} | "
                     f"{roi['diff']:+.6f} |")

    lines.append(f"")

    # --- ROI Level: Largest ADHD+ vs ADHD- absolute differences ---
    lines.append(f"## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences")
    lines.append(f"")
    lines.append(f"ROIs where absolute gradient saliency differs most between classes. "
                 f"Positive diff = more salient for ADHD+.")
    lines.append(f"")

    sorted_by_diff = sorted(saliency_results["roi_saliency"],
                           key=lambda x: abs(x["diff"]), reverse=True)
    lines.append(f"| Rank | ROI | Region | Network | Circuit | Diff | ADHD+ | ADHD- |")
    lines.append(f"|:----:|:---:|--------|---------|---------|:----:|:-----:|:-----:|")
    for rank, roi in enumerate(sorted_by_diff[:20], 1):
        lines.append(f"| {rank} | {roi['roi_idx']} | {roi['region']} | {roi['network']} | "
                     f"{roi['circuit']} | {roi['diff']:+.6f} | "
                     f"{roi['saliency_pos']:.6f} | {roi['saliency_neg']:.6f} |")

    lines.append(f"")

    # --- ROI Level: Signed Saliency (Direction) ---
    lines.append(f"## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)")
    lines.append(f"")
    lines.append(f"Signed saliency shows the direction of each ROI's relationship with ADHD. "
                 f"**Positive** = increasing this ROI's signal pushes prediction toward ADHD+. "
                 f"**Negative** = pushes toward ADHD-.")
    lines.append(f"")

    # Top 10 most positive (pro-ADHD)
    lines.append(f"### Top 10 ROIs with Strongest Positive (+ADHD) Relationship")
    lines.append(f"")
    lines.append(f"| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |")
    lines.append(f"|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|")

    sorted_positive = sorted(saliency_results["roi_saliency"],
                             key=lambda x: x["signed_all"], reverse=True)
    for rank, roi in enumerate(sorted_positive[:10], 1):
        lines.append(f"| {rank} | {roi['roi_idx']} | {roi['region']} | {roi['network']} | "
                     f"{roi['circuit']} | {roi['signed_all']:+.6f} | "
                     f"{roi['signed_pos']:+.6f} | {roi['signed_neg']:+.6f} |")

    lines.append(f"")

    # Top 10 most negative (anti-ADHD)
    lines.append(f"### Top 10 ROIs with Strongest Negative (-ADHD) Relationship")
    lines.append(f"")
    lines.append(f"| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |")
    lines.append(f"|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|")

    sorted_negative = sorted(saliency_results["roi_saliency"],
                             key=lambda x: x["signed_all"])
    for rank, roi in enumerate(sorted_negative[:10], 1):
        lines.append(f"| {rank} | {roi['roi_idx']} | {roi['region']} | {roi['network']} | "
                     f"{roi['circuit']} | {roi['signed_all']:+.6f} | "
                     f"{roi['signed_pos']:+.6f} | {roi['signed_neg']:+.6f} |")

    lines.append(f"")

    report_path = os.path.join(output_dir, "interpretability_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc interpretability analysis for Circuit MoE",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="analysis",
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

    print(f"Model: {model_type}, Circuit: {circuit_config}")

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
    # input_dim from load_abcd_fmri is (N, T, C) = (N, 363, 180)
    # After transpose_fmri_loaders, data is (N, C, T) = (N, 180, 363)
    # but input_dim still reports the original (N, T, C) shape
    n_samples, n_timesteps, n_channels = input_dim
    print(f"Test set: {n_channels} channels, {n_timesteps} timesteps")

    # --- Reconstruct model ---
    if is_v2:
        from models.CircuitMoE_v2 import CircuitMoE
        pool_factor = ckpt_args.get("pool_factor", 10)
        grad_scale = not ckpt_args.get("no_grad_scale", False)
    else:
        from models.CircuitMoE import CircuitMoE
        pool_factor = 1
        grad_scale = False

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
        gating_noise_std=0.0,  # No noise during analysis
        total_channels=n_channels,
        num_classes=ckpt_args.get("num_classes", 2),
        dropout=0.0,  # No dropout during analysis
        device=device,
    )
    if is_v2:
        model_kwargs["pool_factor"] = pool_factor
        model_kwargs["grad_scale"] = False  # No grad scaling during analysis

    model = CircuitMoE(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded. Experts: {model.num_experts} ({', '.join(model.circuit_names)})")

    # --- Run analyses ---
    print("\n" + "=" * 70)
    print("Analysis 1: Gate Weights by Class")
    print("=" * 70)
    gate_results = analyze_gate_weights(model, test_loader, device, model.circuit_names)

    for name, s in gate_results["stats"].items():
        sig = "*" if s["p_value"] < 0.05 else " "
        print(f"  {sig} {name:20s}: ADHD+ {s['adhd_pos_mean']:.4f}, "
              f"ADHD- {s['adhd_neg_mean']:.4f}, "
              f"diff {s['diff']:+.4f}, p={s['p_value']:.4f}")

    print("\n" + "=" * 70)
    print("Analysis 2: Gradient Saliency")
    print("=" * 70)
    saliency_results = analyze_gradient_saliency(
        model, test_loader, device, circuit_config,
    )

    print("\n  Circuit-level saliency (absolute | signed):")
    for name, s in saliency_results["circuit_saliency"].items():
        direction = "+ADHD" if s["signed_all"] > 0 else "-ADHD"
        print(f"    {name:20s}: abs={s['all_mean']:.6f}, "
              f"signed={s['signed_all']:+.6f} ({direction})")

    print("\n  Top 5 networks by absolute saliency:")
    sorted_nets = sorted(saliency_results["network_saliency"].items(),
                        key=lambda x: x[1]["all_mean"], reverse=True)
    for net_name, s in sorted_nets[:5]:
        direction = "+ADHD" if s["signed_all"] > 0 else "-ADHD"
        print(f"    {net_name:20s}: abs={s['all_mean']:.6f}, "
              f"signed={s['signed_all']:+.6f} ({direction})")

    print("\n  Top 5 ROIs by absolute saliency:")
    for roi in saliency_results["roi_saliency"][:5]:
        direction = "+ADHD" if roi["signed_all"] > 0 else "-ADHD"
        print(f"    ROI {roi['roi_idx']:3d} ({roi['network']:20s}, "
              f"{roi['circuit']:15s}): abs={roi['saliency_all']:.6f}, "
              f"signed={roi['signed_all']:+.6f} ({direction})")

    print("\n" + "=" * 70)
    print("Analysis 3: Input Projection Weights")
    print("=" * 70)
    weight_results = analyze_input_weights(model, circuit_config)

    for circuit_name, wr in weight_results.items():
        print(f"\n  Expert: {circuit_name}")
        sorted_nets = sorted(wr["network_summary"].items(),
                            key=lambda x: x[1]["mean_weight_norm"], reverse=True)
        for net_name, ns in sorted_nets:
            print(f"    {net_name:20s}: mean={ns['mean_weight_norm']:.4f}, "
                  f"max={ns['max_weight_norm']:.4f} ({ns['n_rois']} ROIs)")

    # --- Generate report ---
    print("\n" + "=" * 70)
    print("Generating Report")
    print("=" * 70)
    generate_report(
        gate_results, saliency_results, weight_results,
        circuit_config, model_type, args.output_dir,
    )

    # --- Save raw results as JSON ---
    # Separate top ROIs by signed direction for JSON
    sorted_positive = sorted(saliency_results["roi_saliency"],
                             key=lambda x: x["signed_all"], reverse=True)
    sorted_negative = sorted(saliency_results["roi_saliency"],
                             key=lambda x: x["signed_all"])

    json_results = {
        "model_type": model_type,
        "circuit_config": circuit_config,
        "checkpoint": args.checkpoint,
        "gate_weight_stats": gate_results["stats"],
        "circuit_saliency": saliency_results["circuit_saliency"],
        "network_saliency": saliency_results["network_saliency"],
        "roi_saliency_top50": saliency_results["roi_saliency"][:50],
        "roi_signed_top20_positive": sorted_positive[:20],
        "roi_signed_top20_negative": sorted_negative[:20],
        "input_weight_analysis": {
            name: {
                "network_summary": wr["network_summary"],
                "top_rois": wr["top_rois"],
            }
            for name, wr in weight_results.items()
        },
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Raw results saved to: {json_path}")


if __name__ == "__main__":
    main()
