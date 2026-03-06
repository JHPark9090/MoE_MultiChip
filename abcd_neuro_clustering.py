#!/usr/bin/env python3
"""
ABCD fMRI Neurobiology-Based Clustering for Circuit-Specialized MoE.

Clusters subjects using FC features restricted to ADHD-relevant brain
circuits (DMN, Executive, Salience) rather than all 16,110 edges.

Based on abcd_fc_clustering.py but with circuit-specific edge filtering.

Usage:
    python abcd_neuro_clustering.py \
        --circuit-config=adhd_3 \
        --output-dir=./results/neuro_clustering_adhd3 \
        --target-phenotype=ADHD_label \
        --regress-site --use-fisher-z --seed=2025
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from models.yeo17_networks import get_circuit_config, get_circuit_fc_edge_indices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TIMEPOINTS = 363
PARCEL_TYPE = "HCP180"


def _get_fmri_path(data_root, subject_id):
    return os.path.join(
        data_root, f"sub-{subject_id}", f"hcp_mmp1_180_sub-{subject_id}.npy"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subjects(data_root, min_timepoints=300, target_phenotype="ADHD_label"):
    csv_path = os.path.join(data_root, "ABCD_phenotype_total.csv")
    pheno = pd.read_csv(csv_path)

    required = ["subjectkey", target_phenotype]
    optional_demo = ["age", "sex", "race.ethnicity", "abcd_site"]
    keep_cols = required + [c for c in optional_demo
                            if c in pheno.columns and c not in required]
    pheno = pheno[keep_cols].dropna(subset=required).reset_index(drop=True)

    valid = []
    for _, row in pheno.iterrows():
        fpath = _get_fmri_path(data_root, row["subjectkey"])
        if os.path.exists(fpath):
            fmri = np.load(fpath)
            valid.append(fmri.shape[0] >= min_timepoints)
        else:
            valid.append(False)

    pheno = pheno[valid].reset_index(drop=True)
    print(f"[Neuro-Cluster] Valid subjects (T>={min_timepoints}): {len(pheno)}")
    print(f"[Neuro-Cluster] {target_phenotype} distribution:\n"
          f"{pheno[target_phenotype].value_counts().to_string()}")
    return pheno


# ---------------------------------------------------------------------------
# Circuit-specific FC computation
# ---------------------------------------------------------------------------
def compute_circuit_fc_features(pheno, data_root, circuit_config_name,
                                 use_fisher_z=True):
    """Compute FC features restricted to circuit-relevant edges only."""
    n_rois = 180
    circuit_config = get_circuit_config(circuit_config_name)
    edge_indices = get_circuit_fc_edge_indices(circuit_config, n_rois)
    n_circuit_edges = len(edge_indices)

    # Full upper triangle indices
    triu_r, triu_c = np.triu_indices(n_rois, k=1)
    n_all_edges = len(triu_r)

    print(f"[Neuro-Cluster] Circuit config: {circuit_config_name}")
    print(f"[Neuro-Cluster] Circuit edges: {n_circuit_edges} / {n_all_edges} "
          f"({100*n_circuit_edges/n_all_edges:.1f}%)")

    features = np.empty((len(pheno), n_circuit_edges), dtype=np.float32)

    for i, sid in enumerate(pheno["subjectkey"].values):
        fmri = np.load(_get_fmri_path(data_root, sid))
        if fmri.shape[0] > N_TIMEPOINTS:
            start = (fmri.shape[0] - N_TIMEPOINTS) // 2
            fmri = fmri[start:start + N_TIMEPOINTS]

        corr = np.corrcoef(fmri.T)
        np.clip(corr, -1.0, 1.0, out=corr)

        if use_fisher_z:
            np.clip(corr, -0.9999, 0.9999, out=corr)
            corr = np.arctanh(corr)

        # Extract all upper-triangle, then select circuit edges
        all_upper = corr[triu_r, triu_c]
        features[i] = all_upper[edge_indices].astype(np.float32)

        if (i + 1) % 500 == 0 or i == len(pheno) - 1:
            print(f"  ... processed {i + 1}/{len(pheno)} subjects")

    bad = ~np.isfinite(features)
    if bad.any():
        n_bad = bad.any(axis=1).sum()
        warnings.warn(f"{n_bad} subjects have NaN/Inf — replacing with 0")
        features[bad] = 0.0

    return features


# ---------------------------------------------------------------------------
# Site regression
# ---------------------------------------------------------------------------
def regress_out_site(features, site_labels):
    sites = pd.Series(site_labels).fillna(-1).values.reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False, drop="first")
    X_site = enc.fit_transform(sites)
    n_sites = X_site.shape[1] + 1
    print(f"[Site-regress] Regressing {n_sites} sites from {features.shape[1]} features")

    reg = LinearRegression()
    reg.fit(X_site, features)
    residuals = features - reg.predict(X_site)

    ss_total = np.sum((features - features.mean(axis=0, keepdims=True)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r2 = 1.0 - ss_residual / ss_total
    print(f"[Site-regress] R² = {r2:.4f} ({r2*100:.1f}% variance from site)")
    return residuals


# ---------------------------------------------------------------------------
# PCA + Clustering
# ---------------------------------------------------------------------------
def run_pca(features, variance_threshold=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca_full = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum_var, variance_threshold) + 1)
    print(f"[PCA] Components for {variance_threshold*100:.0f}% variance: {n_comp}")

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[PCA] Final: {n_comp} components, cumvar = {cum_var[n_comp-1]:.4f}")
    return X_pca, pca, scaler


def run_clustering(X_pca, max_k=8, seed=2025):
    results = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed, max_iter=500)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)
        results.append({
            "k": k, "labels": labels, "inertia": km.inertia_,
            "silhouette": sil, "calinski": ch, "davies": db,
        })
        print(f"  k={k}: sil={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ABCD Neurobiology-Based Circuit Clustering"
    )
    parser.add_argument("--data-root", type=str,
                        default="/pscratch/sd/j/junghoon/ABCD")
    parser.add_argument("--output-dir", type=str,
                        default="./results/neuro_clustering")
    parser.add_argument("--circuit-config", type=str, default="adhd_3",
                        choices=["adhd_3", "adhd_2"])
    parser.add_argument("--target-phenotype", type=str, default="ADHD_label")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--use-fisher-z", action="store_true")
    parser.add_argument("--regress-site", action="store_true")
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"ABCD Neurobiology-Based Clustering ({args.circuit_config})")
    print("=" * 60)
    print(f"Target: {args.target_phenotype}")
    print(f"Circuit config: {args.circuit_config}")
    print(f"Site regression: {args.regress_site}")
    print()

    # 1. Load subjects
    pheno = load_subjects(args.data_root, target_phenotype=args.target_phenotype)

    # 2. Compute circuit-specific FC
    features = compute_circuit_fc_features(
        pheno, args.data_root, args.circuit_config, args.use_fisher_z,
    )

    # 3. Site regression
    if args.regress_site:
        if "abcd_site" not in pheno.columns:
            raise ValueError("--regress-site requires abcd_site column")
        features = regress_out_site(features, pheno["abcd_site"].values)

    # 4. PCA
    X_pca, pca, scaler = run_pca(features, args.variance_threshold)
    np.save(os.path.join(args.output_dir, "fc_pca_features.npy"), X_pca)

    # 5. Clustering
    print(f"\n[Clustering] KMeans k=2..{args.max_k}")
    results = run_clustering(X_pca, args.max_k, args.seed)

    # 6. Statistical analysis for each k
    phenotype_vals = pheno[args.target_phenotype].values.astype(int)
    pheno_short = args.target_phenotype.replace("_label", "")

    summary_lines = [
        f"Neurobiology-Based Clustering Summary",
        f"Circuit config: {args.circuit_config}",
        f"Target: {args.target_phenotype}",
        f"N subjects: {len(pheno)}",
        f"N circuit edges: {features.shape[1]}",
        f"PCA components: {X_pca.shape[1]}",
        f"Site regression: {args.regress_site}",
        "",
    ]

    best_k = 2
    best_delta = 0.0

    for r in results:
        k = r["k"]
        labels = r["labels"]

        contingency = pd.crosstab(labels, phenotype_vals)
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

        # Phenotype rate per cluster
        rates = []
        for c in range(k):
            mask = labels == c
            rate = phenotype_vals[mask].mean()
            rates.append(rate)
        delta = max(rates) - min(rates)

        line = (f"k={k}: sil={r['silhouette']:.4f}, "
                f"{pheno_short} delta={delta:.4f}, "
                f"chi2={chi2:.2f}, p={p_val:.2e}")
        summary_lines.append(line)
        print(f"  {line}")

        if delta > best_delta:
            best_delta = delta
            best_k = k

    summary_lines.append(f"\nBest k by phenotype delta: {best_k} (delta={best_delta:.4f})")
    print(f"\nBest k by {pheno_short} delta: {best_k} (delta={best_delta:.4f})")

    # 7. Save outputs
    # Save cluster assignments for all k values
    assign_df = pheno.copy()
    for r in results:
        assign_df[f"km_cluster_k{r['k']}"] = r["labels"]
    # Default cluster column uses k=2 (to match existing pipeline)
    k2_result = [r for r in results if r["k"] == 2][0]
    assign_df["km_cluster"] = k2_result["labels"]

    assign_df.to_csv(
        os.path.join(args.output_dir, "cluster_assignments.csv"), index=False
    )

    # Metrics CSV
    metrics_rows = []
    for r in results:
        metrics_rows.append({
            "k": r["k"], "inertia": r["inertia"],
            "silhouette": r["silhouette"],
            "calinski": r["calinski"], "davies": r["davies"],
        })
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(args.output_dir, "clustering_metrics.csv"), index=False
    )

    # Summary
    summary_path = os.path.join(args.output_dir, "clustering_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n[Done] Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
