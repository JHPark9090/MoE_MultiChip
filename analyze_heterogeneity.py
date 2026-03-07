#!/usr/bin/env python3
"""
ADHD heterogeneity analysis using learned representations from Circuit MoE.

Extracts per-subject representations at 3 levels and clusters ADHD+ subjects
to identify potential biotypes:

  Level 1 — Circuit (4D): Gate weight profile per subject
  Level 2 — Network (K*H D): Expert output vectors before gating
  Level 3 — ROI (180D): Input projection activations aggregated per ROI

References:
    - Nigg et al. (2020) Biol. Psychiatry: CNNI — ADHD heterogeneity dimensions
    - Feng et al. (2024) EClinicalMedicine — ADHD biotypes via ABCD dataset
    - Pan et al. (2026) JAMA Psychiatry — ADHD biotypes via morphometry

Usage:
    python analyze_heterogeneity.py \
        --checkpoint=checkpoints/CircuitMoE_classical_adhd_3_49731003.pt \
        --output-dir=analysis/heterogeneity_classical_adhd_3 \
        --n-clusters=3
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
from scipy.stats import f_oneway, kruskal

# scipy.constants must be imported BEFORE pennylane on this system
import scipy.constants  # noqa: F401

from dataloaders.Load_ABCD_fMRI import load_abcd_fmri
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


def load_model_and_data(args):
    """Load checkpoint, reconstruct model, and load test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    model_type = ckpt_args["model_type"]
    circuit_config = ckpt_args["circuit_config"]
    seed = ckpt_args.get("seed", 2025)
    set_all_seeds(seed)

    print(f"Model: {model_type}, Circuit: {circuit_config}")

    is_v2 = "pool_factor" in ckpt_args and ckpt_args["pool_factor"] is not None

    # Load data
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

    # Reconstruct model
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
    model.eval()
    print(f"Model loaded. Experts: {model.num_experts} "
          f"({', '.join(model.circuit_names)})")

    return model, test_loader, device, ckpt_args


# ---------------------------------------------------------------------------
# Level 1: Circuit-Level Representations (Gate Weights)
# ---------------------------------------------------------------------------

def extract_gate_weights(model, test_loader, device):
    """
    Extract per-subject gate weight vectors: (N, K).

    Each subject gets a K-dimensional "circuit fingerprint" showing
    how much the model routes through each brain circuit expert.
    """
    all_gates = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting gate weights", leave=False):
            data, target = batch[0].to(device), batch[1].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            _, gate_weights = model(data)
            all_gates.append(gate_weights.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    gates = np.concatenate(all_gates, axis=0)   # (N, K)
    labels = np.concatenate(all_labels, axis=0)  # (N,)
    return gates, labels


# ---------------------------------------------------------------------------
# Level 2: Network-Level Representations (Expert Output Vectors)
# ---------------------------------------------------------------------------

def extract_expert_outputs(model, test_loader, device):
    """
    Extract per-subject expert output vectors: (N, K, H).

    Each expert produces an H-dimensional hidden vector per subject.
    Concatenated: (N, K*H) gives the full network-level representation.
    These are task-optimized features — better than post-hoc PCA/UMAP
    because the model learned them specifically for ADHD classification.
    """
    all_expert_outs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting expert outputs", leave=False):
            data, target = batch[0].to(device), batch[1].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

            # Manually run the forward pass to capture expert outputs
            B = data.size(0)
            gate_input = data.mean(dim=1)
            gate_weights = model.gate(gate_input)  # (B, K)

            expert_outputs = []
            for i, expert in enumerate(model.experts):
                roi_idx = getattr(model, f"roi_idx_{i}")
                x_subset = data[:, :, roi_idx]

                if model.model_type == "quantum":
                    if model.pool_factor > 1:
                        x_q = x_subset.permute(0, 2, 1)
                        T_trunc = (x_q.size(2) // model.pool_factor) * model.pool_factor
                        x_q = x_q[:, :, :T_trunc]
                        x_q = F.avg_pool1d(x_q, kernel_size=model.pool_factor)
                    else:
                        x_q = x_subset.permute(0, 2, 1)
                    h = expert(x_q)
                else:
                    h = expert(x_subset)

                expert_outputs.append(h)  # each (B, H)

            # Stack: (B, K, H)
            expert_stack = torch.stack(expert_outputs, dim=1)
            all_expert_outs.append(expert_stack.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    expert_outs = np.concatenate(all_expert_outs, axis=0)  # (N, K, H)
    labels = np.concatenate(all_labels, axis=0)
    return expert_outs, labels


# ---------------------------------------------------------------------------
# Level 3: ROI-Level Representations (Input Projection Activations)
# ---------------------------------------------------------------------------

def extract_roi_activations(model, test_loader, device):
    """
    Extract per-subject, per-ROI activation magnitudes: (N, 180).

    For each expert, capture the output of the input_projection layer
    (or feature_projection for quantum), then map back to global ROI indices
    using the weight matrix norms as a "contribution score".

    For each subject:
      roi_score[global_idx] += ||W[:, local_idx]||_2 * ||activation[t, :]||_2

    This gives a 180-D vector showing how much each ROI contributes to
    the model's internal representation for that subject.
    """
    config_name = None
    # Infer config from model's circuit_names
    for cfg_name in ["adhd_3", "adhd_2", "arbitrary_4", "arbitrary_2"]:
        try:
            cfg = get_circuit_config(cfg_name)
            roi_list = get_circuit_roi_indices(cfg)
            names = [n for n, _ in roi_list]
            if names == model.circuit_names:
                config_name = cfg_name
                break
        except (KeyError, ValueError):
            continue

    if config_name is None:
        print("  Warning: Could not infer circuit config. Skipping ROI-level analysis.")
        return None, None, None

    config = get_circuit_config(config_name)
    circuit_roi_list = get_circuit_roi_indices(config)

    # Pre-extract weight norms per expert
    expert_weight_norms = []
    for i, expert in enumerate(model.experts):
        if hasattr(expert, 'input_projection'):
            W = expert.input_projection.weight.detach().cpu().numpy()  # (H, n_rois)
        elif hasattr(expert, 'feature_projection'):
            W = expert.feature_projection.weight.detach().cpu().numpy()  # (n_rots, n_rois)
        else:
            expert_weight_norms.append(None)
            continue
        # L2 norm per input column -> (n_rois,)
        expert_weight_norms.append(np.linalg.norm(W, axis=0))

    all_roi_scores = []
    all_signed_roi_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting ROI activations", leave=False):
            data, target = batch[0].to(device), batch[1].to(device)
            data = data.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            B = data.size(0)

            roi_scores = np.zeros((B, 180), dtype=np.float32)
            signed_roi_scores = np.zeros((B, 180), dtype=np.float32)

            for i, expert in enumerate(model.experts):
                roi_idx = getattr(model, f"roi_idx_{i}")
                x_subset = data[:, :, roi_idx]  # (B, T, n_rois_i)

                if expert_weight_norms[i] is None:
                    continue

                w_norms = expert_weight_norms[i]  # (n_rois_i,)

                # Absolute importance: |input| * weight_norm (magnitude only)
                input_magnitude = x_subset.abs().mean(dim=1).cpu().numpy()  # (B, n_rois_i)
                roi_importance = input_magnitude * w_norms[np.newaxis, :]  # (B, n_rois_i)

                # Signed importance: input_mean * weight_norm (preserves direction)
                # Positive = this ROI has positive mean activation scaled by weight
                # Negative = this ROI has negative mean activation scaled by weight
                input_signed = x_subset.mean(dim=1).cpu().numpy()  # (B, n_rois_i)
                roi_signed = input_signed * w_norms[np.newaxis, :]  # (B, n_rois_i)

                # Map back to global ROI indices
                global_indices = circuit_roi_list[i][1]
                for local_idx, global_idx in enumerate(global_indices):
                    roi_scores[:, global_idx] = roi_importance[:, local_idx]
                    signed_roi_scores[:, global_idx] = roi_signed[:, local_idx]

            all_roi_scores.append(roi_scores)
            all_signed_roi_scores.append(signed_roi_scores)
            all_labels.append(target.cpu().numpy())

    roi_scores = np.concatenate(all_roi_scores, axis=0)  # (N, 180)
    signed_roi_scores = np.concatenate(all_signed_roi_scores, axis=0)  # (N, 180)
    labels = np.concatenate(all_labels, axis=0)
    return roi_scores, signed_roi_scores, labels


# ---------------------------------------------------------------------------
# Clustering and Characterization
# ---------------------------------------------------------------------------

def cluster_representations(representations, n_clusters, level_name):
    """
    Cluster subjects using k-means. Returns cluster assignments and stats.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=2025)
    cluster_ids = km.fit_predict(representations)

    sil = silhouette_score(representations, cluster_ids) if n_clusters > 1 else 0.0

    print(f"\n  {level_name} clustering (K={n_clusters}):")
    print(f"    Silhouette score: {sil:.4f}")
    for c in range(n_clusters):
        n = (cluster_ids == c).sum()
        print(f"    Cluster {c}: {n} subjects ({100*n/len(cluster_ids):.1f}%)")

    return cluster_ids, km.cluster_centers_, sil


def characterize_circuit_clusters(gate_weights, cluster_ids, circuit_names,
                                  n_clusters):
    """
    Characterize each cluster by its mean gate weight profile.
    Identifies the dominant circuit for each cluster.
    """
    results = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        mean_gates = gate_weights[mask].mean(axis=0)
        std_gates = gate_weights[mask].std(axis=0)
        dominant_idx = int(np.argmax(mean_gates))

        cluster_info = {
            "n_subjects": int(mask.sum()),
            "dominant_circuit": circuit_names[dominant_idx],
            "gate_profile": {},
        }
        for i, name in enumerate(circuit_names):
            cluster_info["gate_profile"][name] = {
                "mean": float(mean_gates[i]),
                "std": float(std_gates[i]),
            }
        results[f"cluster_{c}"] = cluster_info

    # Statistical test: do gate weights differ across clusters?
    gate_stats = {}
    for i, name in enumerate(circuit_names):
        groups = [gate_weights[cluster_ids == c, i] for c in range(n_clusters)]
        if n_clusters >= 2:
            f_stat, p_val = f_oneway(*groups)
            gate_stats[name] = {"F_stat": float(f_stat), "p_value": float(p_val)}
        else:
            gate_stats[name] = {"F_stat": 0.0, "p_value": 1.0}

    results["gate_anova"] = gate_stats
    return results


def characterize_expert_clusters(expert_outputs, cluster_ids, circuit_names,
                                 n_clusters):
    """
    Characterize each cluster by mean expert output norms.
    Shows which experts produce stronger activations per cluster.
    """
    K, H = expert_outputs.shape[1], expert_outputs.shape[2]
    results = {}

    for c in range(n_clusters):
        mask = cluster_ids == c
        cluster_outs = expert_outputs[mask]  # (n_c, K, H)

        # Per-expert L2 norm: mean over subjects
        expert_norms = np.linalg.norm(cluster_outs, axis=2).mean(axis=0)  # (K,)
        dominant_idx = int(np.argmax(expert_norms))

        cluster_info = {
            "n_subjects": int(mask.sum()),
            "dominant_expert": circuit_names[dominant_idx],
            "expert_norms": {},
        }
        for i, name in enumerate(circuit_names):
            cluster_info["expert_norms"][name] = float(expert_norms[i])

        results[f"cluster_{c}"] = cluster_info

    return results


def characterize_roi_clusters(roi_scores, signed_roi_scores, cluster_ids,
                              n_clusters):
    """
    Characterize each cluster by top ROIs, network composition, and
    signed direction (positive/negative relationship with ADHD).
    """
    roi_to_net = {}
    for net_name, indices in YEO17_HCP180.items():
        for idx in indices:
            roi_to_net[idx] = net_name

    results = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        mean_scores = roi_scores[mask].mean(axis=0)  # (180,)
        mean_signed = signed_roi_scores[mask].mean(axis=0)  # (180,)
        top_roi_indices = np.argsort(mean_scores)[::-1][:10]

        top_rois = []
        for roi_idx in top_roi_indices:
            top_rois.append({
                "roi_idx": int(roi_idx),
                "network": roi_to_net.get(int(roi_idx), "unknown"),
                "score": float(mean_scores[roi_idx]),
                "signed_score": float(mean_signed[roi_idx]),
                "direction": "+ADHD" if mean_signed[roi_idx] > 0 else "-ADHD",
            })

        # Top 5 most positive and most negative ROIs
        top_positive_idx = np.argsort(mean_signed)[::-1][:5]
        top_negative_idx = np.argsort(mean_signed)[:5]

        top_positive = []
        for roi_idx in top_positive_idx:
            top_positive.append({
                "roi_idx": int(roi_idx),
                "network": roi_to_net.get(int(roi_idx), "unknown"),
                "signed_score": float(mean_signed[roi_idx]),
            })

        top_negative = []
        for roi_idx in top_negative_idx:
            top_negative.append({
                "roi_idx": int(roi_idx),
                "network": roi_to_net.get(int(roi_idx), "unknown"),
                "signed_score": float(mean_signed[roi_idx]),
            })

        # Network-level aggregation (both absolute and signed)
        network_scores = {}
        for net_name, indices in YEO17_HCP180.items():
            network_scores[net_name] = {
                "absolute": float(mean_scores[indices].mean()),
                "signed": float(mean_signed[indices].mean()),
                "direction": "+ADHD" if mean_signed[indices].mean() > 0 else "-ADHD",
            }

        results[f"cluster_{c}"] = {
            "n_subjects": int(mask.sum()),
            "top_rois": top_rois,
            "top_positive_rois": top_positive,
            "top_negative_rois": top_negative,
            "network_scores": network_scores,
        }

    return results


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_heterogeneity_report(circuit_results, expert_results,
                                  roi_results, circuit_names, model_type,
                                  circuit_config, n_clusters, silhouettes,
                                  output_dir):
    """Generate markdown report for heterogeneity analysis."""
    lines = []
    lines.append("# ADHD Heterogeneity Analysis via Learned Representations")
    lines.append("")
    lines.append(f"**Model**: {model_type} | **Config**: {circuit_config} | "
                 f"**K**: {n_clusters} clusters")
    lines.append("")

    # --- Level 1: Circuit ---
    lines.append("## Level 1: Circuit-Level Heterogeneity (Gate Weights)")
    lines.append("")
    lines.append("Each ADHD+ subject has a K-dimensional gate weight vector — "
                 "a 'circuit fingerprint' showing how the model routes their brain "
                 "data through DMN, Executive, Salience, and SensoriMotor experts.")
    lines.append("")
    lines.append(f"**Silhouette score**: {silhouettes['circuit']:.4f}")
    lines.append("")

    # Cluster profiles
    lines.append("### Cluster Profiles")
    lines.append("")
    header = "| Cluster | N | Dominant Circuit |"
    sep = "|:-------:|:-:|:----------------:|"
    for name in circuit_names:
        header += f" {name} |"
        sep += ":-----:|"
    lines.append(header)
    lines.append(sep)

    for c in range(n_clusters):
        ci = circuit_results[f"cluster_{c}"]
        row = f"| {c} | {ci['n_subjects']} | {ci['dominant_circuit']} |"
        for name in circuit_names:
            g = ci["gate_profile"][name]
            row += f" {g['mean']:.3f}±{g['std']:.3f} |"
        lines.append(row)
    lines.append("")

    # ANOVA
    lines.append("### Gate Weight ANOVA Across Clusters")
    lines.append("")
    lines.append("| Circuit | F-stat | p-value | Significant? |")
    lines.append("|---------|:------:|:-------:|:------------:|")
    for name in circuit_names:
        gs = circuit_results["gate_anova"][name]
        sig = "Yes" if gs["p_value"] < 0.05 else "No"
        lines.append(f"| {name} | {gs['F_stat']:.3f} | {gs['p_value']:.4f} | {sig} |")
    lines.append("")

    # --- Level 2: Network (Expert Output) ---
    lines.append("## Level 2: Network-Level Heterogeneity (Expert Outputs)")
    lines.append("")
    lines.append("Expert output vectors (K×H dimensions) capture task-optimized "
                 "representations learned by each circuit expert. These are richer "
                 "than gate weights — they encode *what* each expert learned, not "
                 "just *how much* it was used.")
    lines.append("")
    lines.append(f"**Silhouette score**: {silhouettes['network']:.4f}")
    lines.append("")

    lines.append("### Expert Activation Norms per Cluster")
    lines.append("")
    header = "| Cluster | N | Dominant Expert |"
    sep = "|:-------:|:-:|:---------------:|"
    for name in circuit_names:
        header += f" {name} |"
        sep += ":-----:|"
    lines.append(header)
    lines.append(sep)

    for c in range(n_clusters):
        ci = expert_results[f"cluster_{c}"]
        row = f"| {c} | {ci['n_subjects']} | {ci['dominant_expert']} |"
        for name in circuit_names:
            row += f" {ci['expert_norms'][name]:.3f} |"
        lines.append(row)
    lines.append("")

    # --- Level 3: ROI ---
    if roi_results is not None:
        lines.append("## Level 3: ROI-Level Heterogeneity (Input Projection Activations)")
        lines.append("")
        lines.append("Per-ROI importance scores combine input magnitude with learned "
                     "weight norms — revealing which of the 180 brain regions are most "
                     "important for each ADHD subtype cluster.")
        lines.append("")
        lines.append(f"**Silhouette score**: {silhouettes['roi']:.4f}")
        lines.append("")

        for c in range(n_clusters):
            ci = roi_results[f"cluster_{c}"]
            lines.append(f"### Cluster {c} — Top 10 ROIs by Importance (N={ci['n_subjects']})")
            lines.append("")
            lines.append("| Rank | ROI | Network | Importance | Signed | Direction |")
            lines.append("|:----:|:---:|---------|:----------:|:------:|:---------:|")
            for rank, roi in enumerate(ci["top_rois"], 1):
                lines.append(f"| {rank} | {roi['roi_idx']} | "
                             f"{roi['network']} | {roi['score']:.4f} | "
                             f"{roi['signed_score']:+.4f} | {roi['direction']} |")
            lines.append("")

            # Positive and negative ROIs per cluster
            lines.append(f"**Top 5 ROIs with positive (+ADHD) relationship:**")
            lines.append("")
            lines.append("| ROI | Network | Signed Score |")
            lines.append("|:---:|---------|:------------:|")
            for roi in ci["top_positive_rois"]:
                lines.append(f"| {roi['roi_idx']} | {roi['network']} | "
                             f"{roi['signed_score']:+.4f} |")
            lines.append("")

            lines.append(f"**Top 5 ROIs with negative (-ADHD) relationship:**")
            lines.append("")
            lines.append("| ROI | Network | Signed Score |")
            lines.append("|:---:|---------|:------------:|")
            for roi in ci["top_negative_rois"]:
                lines.append(f"| {roi['roi_idx']} | {roi['network']} | "
                             f"{roi['signed_score']:+.4f} |")
            lines.append("")

        # Cross-cluster network comparison (absolute + signed)
        lines.append("### Network-Level Scores per Cluster (Absolute)")
        lines.append("")
        all_nets = sorted(YEO17_HCP180.keys())
        header = "| Network |"
        sep = "|---------|"
        for c in range(n_clusters):
            header += f" Cluster {c} |"
            sep += ":--------:|"
        lines.append(header)
        lines.append(sep)
        for net in all_nets:
            row = f"| {net} |"
            for c in range(n_clusters):
                score = roi_results[f"cluster_{c}"]["network_scores"][net]["absolute"]
                row += f" {score:.4f} |"
            lines.append(row)
        lines.append("")

        lines.append("### Network-Level Signed Scores per Cluster (Direction)")
        lines.append("")
        lines.append("Positive = network positively associated with ADHD in this cluster. "
                     "Negative = negatively associated.")
        lines.append("")
        header = "| Network |"
        sep = "|---------|"
        for c in range(n_clusters):
            header += f" Cluster {c} |"
            sep += ":--------:|"
        lines.append(header)
        lines.append(sep)
        for net in all_nets:
            row = f"| {net} |"
            for c in range(n_clusters):
                score = roi_results[f"cluster_{c}"]["network_scores"][net]["signed"]
                direction = roi_results[f"cluster_{c}"]["network_scores"][net]["direction"]
                row += f" {score:+.4f} ({direction}) |"
            lines.append(row)
        lines.append("")

    # --- Caveats ---
    lines.append("## Caveats")
    lines.append("")
    lines.append("1. **Model performance ceiling**: AUC ~0.58-0.62. "
                 "Subtype structure from a near-chance model is exploratory.")
    lines.append("2. **Load balancing suppresses routing differences**: "
                 "Auxiliary loss encourages uniform routing, compressing "
                 "circuit-level heterogeneity.")
    lines.append("3. **Single seed**: Results from seed=2025 only. "
                 "Cluster stability across seeds is not validated.")
    lines.append("4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. "
                 "Subcortical structures (caudate, pallidum) are absent.")
    lines.append("5. **Cluster count**: K chosen a priori. "
                 "Silhouette scores should guide optimal K selection.")
    lines.append("")

    report_path = os.path.join(output_dir, "heterogeneity_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="ADHD heterogeneity analysis via Circuit MoE learned representations",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained Circuit MoE checkpoint")
    parser.add_argument("--output-dir", type=str, default="analysis/heterogeneity",
                        help="Directory for analysis outputs")
    parser.add_argument("--n-clusters", type=int, default=3,
                        help="Number of clusters for k-means (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-k", type=int, default=6,
                        help="Max K to test for silhouette sweep (default: 6)")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, test_loader, device, ckpt_args = load_model_and_data(args)
    circuit_names = model.circuit_names
    circuit_config = ckpt_args["circuit_config"]
    model_type = ckpt_args["model_type"]

    # ---------------------------------------------------------------
    # Extract representations at all 3 levels
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Extracting per-subject representations")
    print("=" * 70)

    gate_weights, labels = extract_gate_weights(model, test_loader, device)
    print(f"  Gate weights: {gate_weights.shape}")

    expert_outputs, _ = extract_expert_outputs(model, test_loader, device)
    print(f"  Expert outputs: {expert_outputs.shape}")

    roi_scores, signed_roi_scores, _ = extract_roi_activations(
        model, test_loader, device,
    )
    if roi_scores is not None:
        print(f"  ROI scores: {roi_scores.shape}")

    # ---------------------------------------------------------------
    # Focus on ADHD+ subjects for subtyping
    # ---------------------------------------------------------------
    adhd_mask = labels == 1
    n_adhd = adhd_mask.sum()
    print(f"\nADHD+ subjects for clustering: {n_adhd}")

    if n_adhd < args.n_clusters * 2:
        print(f"ERROR: Too few ADHD+ subjects ({n_adhd}) for "
              f"{args.n_clusters} clusters. Aborting.")
        return

    gates_adhd = gate_weights[adhd_mask]
    experts_adhd = expert_outputs[adhd_mask]
    experts_adhd_flat = experts_adhd.reshape(n_adhd, -1)  # (N_adhd, K*H)
    roi_adhd = roi_scores[adhd_mask] if roi_scores is not None else None
    signed_roi_adhd = signed_roi_scores[adhd_mask] if signed_roi_scores is not None else None

    # ---------------------------------------------------------------
    # Silhouette sweep to find optimal K
    # ---------------------------------------------------------------
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print("\n" + "=" * 70)
    print("Silhouette Sweep (Circuit-Level)")
    print("=" * 70)

    sil_scores = {}
    for k in range(2, min(args.max_k + 1, n_adhd)):
        km = KMeans(n_clusters=k, n_init=10, random_state=2025)
        cids = km.fit_predict(gates_adhd)
        sil = silhouette_score(gates_adhd, cids)
        sil_scores[k] = sil
        print(f"  K={k}: silhouette={sil:.4f}")

    best_k_auto = max(sil_scores, key=sil_scores.get) if sil_scores else args.n_clusters
    print(f"  Best K by silhouette: {best_k_auto}")

    # Use user-specified K for the main analysis
    n_clusters = args.n_clusters
    print(f"\nUsing K={n_clusters} for main analysis (use --n-clusters to change)")

    # ---------------------------------------------------------------
    # Cluster at each level
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Clustering ADHD+ Subjects (K={n_clusters})")
    print("=" * 70)

    # Level 1: Circuit
    circuit_cluster_ids, circuit_centers, sil_circuit = cluster_representations(
        gates_adhd, n_clusters, "Circuit (gate weights)"
    )

    # Level 2: Network
    network_cluster_ids, network_centers, sil_network = cluster_representations(
        experts_adhd_flat, n_clusters, "Network (expert outputs)"
    )

    # Level 3: ROI
    if roi_adhd is not None:
        roi_cluster_ids, roi_centers, sil_roi = cluster_representations(
            roi_adhd, n_clusters, "ROI (projection activations)"
        )
    else:
        roi_cluster_ids, sil_roi = None, 0.0

    # ---------------------------------------------------------------
    # Characterize clusters
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Characterizing Clusters")
    print("=" * 70)

    circuit_results = characterize_circuit_clusters(
        gates_adhd, circuit_cluster_ids, circuit_names, n_clusters
    )

    expert_results = characterize_expert_clusters(
        experts_adhd, network_cluster_ids, circuit_names, n_clusters
    )

    roi_results = None
    if roi_adhd is not None:
        roi_results = characterize_roi_clusters(
            roi_adhd, signed_roi_adhd, roi_cluster_ids, n_clusters
        )

    # Print circuit cluster characterization
    print("\n  Circuit-level cluster profiles:")
    for c in range(n_clusters):
        ci = circuit_results[f"cluster_{c}"]
        profile = ", ".join(
            f"{name}={ci['gate_profile'][name]['mean']:.3f}"
            for name in circuit_names
        )
        print(f"    Cluster {c} (N={ci['n_subjects']}, "
              f"dominant={ci['dominant_circuit']}): {profile}")

    # ---------------------------------------------------------------
    # Cluster concordance across levels
    # ---------------------------------------------------------------
    from sklearn.metrics import adjusted_rand_score

    print("\n  Cross-level concordance (Adjusted Rand Index):")
    ari_cn = adjusted_rand_score(circuit_cluster_ids, network_cluster_ids)
    print(f"    Circuit vs Network: ARI = {ari_cn:.4f}")
    if roi_cluster_ids is not None:
        ari_cr = adjusted_rand_score(circuit_cluster_ids, roi_cluster_ids)
        ari_nr = adjusted_rand_score(network_cluster_ids, roi_cluster_ids)
        print(f"    Circuit vs ROI:     ARI = {ari_cr:.4f}")
        print(f"    Network vs ROI:     ARI = {ari_nr:.4f}")
    else:
        ari_cr, ari_nr = 0.0, 0.0

    # ---------------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generating Report")
    print("=" * 70)

    silhouettes = {
        "circuit": sil_circuit,
        "network": sil_network,
        "roi": sil_roi,
    }

    generate_heterogeneity_report(
        circuit_results, expert_results, roi_results,
        circuit_names, model_type, circuit_config,
        n_clusters, silhouettes, args.output_dir,
    )

    # ---------------------------------------------------------------
    # Save raw results
    # ---------------------------------------------------------------
    json_results = {
        "model_type": model_type,
        "circuit_config": circuit_config,
        "checkpoint": args.checkpoint,
        "n_clusters": n_clusters,
        "n_adhd_subjects": int(n_adhd),
        "silhouette_sweep": {str(k): float(v) for k, v in sil_scores.items()},
        "best_k_auto": int(best_k_auto),
        "silhouettes": {k: float(v) for k, v in silhouettes.items()},
        "cross_level_ari": {
            "circuit_vs_network": float(ari_cn),
            "circuit_vs_roi": float(ari_cr),
            "network_vs_roi": float(ari_nr),
        },
        "circuit_clusters": circuit_results,
        "expert_clusters": expert_results,
    }
    if roi_results is not None:
        json_results["roi_clusters"] = roi_results

    json_path = os.path.join(args.output_dir, "heterogeneity_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Raw results saved to: {json_path}")

    # Save per-subject data as .npz for downstream analysis
    npz_path = os.path.join(args.output_dir, "subject_representations.npz")
    save_dict = {
        "gate_weights": gate_weights,
        "expert_outputs": expert_outputs,
        "labels": labels,
        "circuit_cluster_ids": circuit_cluster_ids,
        "network_cluster_ids": network_cluster_ids,
    }
    if roi_scores is not None:
        save_dict["roi_scores"] = roi_scores
    if signed_roi_scores is not None:
        save_dict["signed_roi_scores"] = signed_roi_scores
    if roi_cluster_ids is not None:
        save_dict["roi_cluster_ids"] = roi_cluster_ids
    np.savez(npz_path, **save_dict)
    print(f"Subject representations saved to: {npz_path}")


if __name__ == "__main__":
    main()
