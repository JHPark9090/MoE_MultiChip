#!/usr/bin/env python3
"""
ABCD fMRI Functional Connectivity Clustering Analysis
=====================================================

Clusters ABCD subjects based on their resting-state connectivity patterns
to identify neural subtypes for MoE expert selection.

Supports two feature types:
  - **fc** (default): Pearson correlation FC matrices (+ optional Fisher-z)
  - **coherence**: Band-averaged magnitude-squared coherence matrices
    (frequency-domain connectivity in a specified band, default 0.01-0.1 Hz)

Pipeline:
  1. Load phenotype + fMRI data (HCP180 parcellation)
  2. Compute per-subject connectivity matrices (FC or coherence)
  3. PCA dimensionality reduction on upper-triangle features
  4. KMeans / Spectral / Hierarchical clustering across k=2..max_k
  5. Visualisation (10 plots) and statistical analysis
  6. Save cluster assignments, metrics, and PCA features

Usage:
  # FC-based (default):
  python abcd_fc_clustering.py \
      --data-root=/pscratch/sd/j/junghoon/ABCD \
      --output-dir=./results/fc_clustering \
      --seed=2025 --use-fisher-z

  # Coherence-based:
  python abcd_fc_clustering.py \
      --feature-type=coherence \
      --tr=0.8 --freq-band 0.01 0.1 \
      --output-dir=./results/coherence_clustering \
      --seed=2025
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Constants (mirrored from dataloaders/Load_ABCD_fMRI.py)
# ---------------------------------------------------------------------------
N_TIMEPOINTS = 363
PARCEL_DIMS = {"HCP": 360, "HCP180": 180, "Schaefer": 400}
PARCEL_TYPE = "HCP180"


def _get_fmri_path(data_root, subject_id, parcel_type=PARCEL_TYPE):
    """Return the fMRI .npy path for a given subject."""
    if parcel_type == "HCP":
        fname = f"hcp_mmp1_sub-{subject_id}.npy"
    elif parcel_type == "HCP180":
        fname = f"hcp_mmp1_180_sub-{subject_id}.npy"
    elif parcel_type == "Schaefer":
        fname = f"schaefer_sub-{subject_id}.npy"
    else:
        raise ValueError(f"Unknown parcel_type: {parcel_type!r}")
    return os.path.join(data_root, f"sub-{subject_id}", fname)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_subjects(data_root, min_timepoints=300):
    """Load phenotype CSV and filter to subjects with valid fMRI on disk."""
    csv_path = os.path.join(data_root, "ABCD_phenotype_total.csv")
    pheno = pd.read_csv(csv_path)

    required = ["subjectkey", "ADHD_label"]
    optional_demo = ["age", "sex", "race.ethnicity", "abcd_site"]
    keep_cols = required + [c for c in optional_demo if c in pheno.columns]
    pheno = pheno[keep_cols].dropna(subset=required).reset_index(drop=True)

    # Check which subjects have valid fMRI files
    valid = []
    for _, row in pheno.iterrows():
        sid = row["subjectkey"]
        fpath = _get_fmri_path(data_root, sid)
        if os.path.exists(fpath):
            fmri = np.load(fpath)
            if fmri.shape[0] >= min_timepoints:
                valid.append(True)
            else:
                valid.append(False)
        else:
            valid.append(False)

    pheno = pheno[valid].reset_index(drop=True)
    print(f"[FC-Cluster] Valid subjects (T>={min_timepoints}): {len(pheno)}")
    print(f"[FC-Cluster] ADHD label distribution:\n{pheno['ADHD_label'].value_counts().to_string()}")
    return pheno


# ---------------------------------------------------------------------------
# 2. FC matrix computation
# ---------------------------------------------------------------------------
def compute_fc_features(pheno, data_root, use_fisher_z=True):
    """Compute upper-triangle FC features for each subject.

    Returns:
        fc_features: (N, n_features) array  — upper triangle of FC matrix
        n_rois: number of ROIs
    """
    n_rois = PARCEL_DIMS[PARCEL_TYPE]
    triu_idx = np.triu_indices(n_rois, k=1)
    n_features = len(triu_idx[0])

    print(f"[FC-Cluster] Computing FC matrices ({n_rois} ROIs, {n_features} upper-tri features)...")

    fc_features = np.empty((len(pheno), n_features), dtype=np.float32)

    for i, sid in enumerate(pheno["subjectkey"].values):
        fmri = np.load(_get_fmri_path(data_root, sid))  # (T, R)

        # Center-crop to N_TIMEPOINTS if longer
        if fmri.shape[0] > N_TIMEPOINTS:
            start = (fmri.shape[0] - N_TIMEPOINTS) // 2
            fmri = fmri[start : start + N_TIMEPOINTS]

        # Pearson correlation: corrcoef expects (variables, observations)
        corr = np.corrcoef(fmri.T)  # (R, R)

        # Clip numerical noise outside [-1, 1] before arctanh
        np.clip(corr, -1.0, 1.0, out=corr)

        if use_fisher_z:
            # Clip away from exactly +/-1 to avoid inf in arctanh
            np.clip(corr, -0.9999, 0.9999, out=corr)
            corr = np.arctanh(corr)

        fc_features[i] = corr[triu_idx].astype(np.float32)

        if (i + 1) % 500 == 0 or i == len(pheno) - 1:
            print(f"  ... processed {i + 1}/{len(pheno)} subjects")

    # Check for NaN/Inf
    bad = ~np.isfinite(fc_features)
    if bad.any():
        n_bad = bad.any(axis=1).sum()
        warnings.warn(f"{n_bad} subjects have NaN/Inf in FC features — replacing with 0")
        fc_features[bad] = 0.0

    return fc_features, n_rois


# ---------------------------------------------------------------------------
# 2b. Coherence matrix computation
# ---------------------------------------------------------------------------
def _load_and_crop(data_root, sid):
    """Load fMRI and center-crop to N_TIMEPOINTS."""
    fmri = np.load(_get_fmri_path(data_root, sid))  # (T, R)
    if fmri.shape[0] > N_TIMEPOINTS:
        start = (fmri.shape[0] - N_TIMEPOINTS) // 2
        fmri = fmri[start : start + N_TIMEPOINTS]
    return fmri


def _coherence_matrix(fmri, fs, freq_band, nperseg=None):
    """Compute band-averaged magnitude-squared coherence for all ROI pairs.

    Uses a vectorized Welch-like approach: segment the data, FFT each segment,
    accumulate cross-spectral and auto-spectral density, then normalise.

    Args:
        fmri: (T, R) array — time series for one subject.
        fs: float — sampling frequency in Hz (1/TR).
        freq_band: (f_low, f_high) — frequency band for averaging.
        nperseg: int or None — Welch segment length (default: T//2).

    Returns:
        coh: (R, R) array — coherence values in [0, 1].
    """
    T, R = fmri.shape
    if nperseg is None:
        nperseg = min(T // 2, 128)
    noverlap = nperseg // 2
    step = nperseg - noverlap

    # Hann window
    window = np.hanning(nperseg).astype(np.float64)
    win_norm = np.sum(window ** 2)  # for proper PSD scaling

    # Frequency axis
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    n_band = int(band_mask.sum())
    if n_band == 0:
        warnings.warn(
            f"No frequency bins in band [{freq_band[0]}, {freq_band[1]}] Hz "
            f"with nperseg={nperseg}, fs={fs:.3f} Hz. Using all bins."
        )
        band_mask = np.ones(len(freqs), dtype=bool)
        n_band = len(freqs)

    # Segment boundaries
    starts = list(range(0, T - nperseg + 1, step))
    n_seg = len(starts)
    if n_seg == 0:
        # Fallback: use full signal as single segment
        starts = [0]
        nperseg = T
        n_seg = 1
        window = np.hanning(nperseg).astype(np.float64)
        win_norm = np.sum(window ** 2)
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        n_band = int(band_mask.sum())
        if n_band == 0:
            band_mask = np.ones(len(freqs), dtype=bool)
            n_band = len(freqs)

    # Accumulate cross- and auto-spectral densities
    Sxy = np.zeros((R, R), dtype=np.complex128)
    Sxx = np.zeros(R, dtype=np.float64)

    for s in starts:
        seg = fmri[s : s + nperseg].astype(np.float64) * window[:, None]  # (nperseg, R)
        F = np.fft.rfft(seg, axis=0)  # (n_freqs, R)
        F_band = F[band_mask]  # (n_band, R)

        # Cross-spectral: sum_f F[f,:] @ F[f,:]^H  → (R, R)
        Sxy += np.einsum("fi,fj->ij", F_band, F_band.conj())
        # Auto-spectral
        Sxx += np.sum(np.abs(F_band) ** 2, axis=0)

    # Magnitude-squared coherence: |Sxy|^2 / (Sxx_i * Sxx_j)
    denom = np.outer(Sxx, Sxx)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    coh = np.abs(Sxy) ** 2 / denom

    # Clip to [0, 1] for numerical safety
    np.clip(coh, 0.0, 1.0, out=coh)

    return coh


def compute_coherence_features(pheno, data_root, tr=0.8, freq_band=(0.01, 0.1),
                                use_fisher_z=False):
    """Compute upper-triangle coherence features for each subject.

    Args:
        pheno: DataFrame with 'subjectkey' column.
        data_root: ABCD data directory.
        tr: Repetition time in seconds.
        freq_band: (f_low, f_high) Hz for band-averaging.
        use_fisher_z: If True, apply arctanh to coherence values.

    Returns:
        features: (N, n_features) array — upper triangle of coherence matrix.
        n_rois: number of ROIs.
    """
    fs = 1.0 / tr
    n_rois = PARCEL_DIMS[PARCEL_TYPE]
    triu_idx = np.triu_indices(n_rois, k=1)
    n_features = len(triu_idx[0])

    print(f"[Coherence] Computing coherence matrices "
          f"({n_rois} ROIs, {n_features} upper-tri features)")
    print(f"[Coherence] TR={tr}s, fs={fs:.3f} Hz, band={freq_band}")

    features = np.empty((len(pheno), n_features), dtype=np.float32)

    for i, sid in enumerate(pheno["subjectkey"].values):
        fmri = _load_and_crop(data_root, sid)
        coh = _coherence_matrix(fmri, fs, freq_band)

        if use_fisher_z:
            # Coherence is in [0,1]; arctanh needs input in (-1,1)
            np.clip(coh, 0.0001, 0.9999, out=coh)
            coh = np.arctanh(coh)

        features[i] = coh[triu_idx].astype(np.float32)

        if (i + 1) % 500 == 0 or i == len(pheno) - 1:
            print(f"  ... processed {i + 1}/{len(pheno)} subjects")

    # Check for NaN/Inf
    bad = ~np.isfinite(features)
    if bad.any():
        n_bad = bad.any(axis=1).sum()
        warnings.warn(f"{n_bad} subjects have NaN/Inf in coherence features — replacing with 0")
        features[bad] = 0.0

    return features, n_rois


# ---------------------------------------------------------------------------
# 3. PCA dimensionality reduction
# ---------------------------------------------------------------------------
def run_pca(fc_features, variance_threshold=0.95, n_components_override=None):
    """Standardise and run PCA. Returns transformed features + fitted PCA."""
    scaler = StandardScaler()
    fc_scaled = scaler.fit_transform(fc_features)

    # First pass: fit full PCA to determine n_components
    if n_components_override is not None and n_components_override > 0:
        n_comp = min(n_components_override, fc_scaled.shape[0], fc_scaled.shape[1])
    else:
        pca_full = PCA(n_components=min(fc_scaled.shape[0], fc_scaled.shape[1]))
        pca_full.fit(fc_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum_var, variance_threshold) + 1)
        print(f"[PCA] Components for {variance_threshold*100:.0f}% variance: {n_comp}")

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(fc_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"[PCA] Final: {n_comp} components, cumulative variance = {cum_var[-1]:.4f}")
    return X_pca, pca, scaler


# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------
def run_clustering(X_pca, max_k=8, seed=2025):
    """Run KMeans, Spectral, and Agglomerative clustering for k=2..max_k."""
    results = []
    kmeans_models = {}

    for k in range(2, max_k + 1):
        row = {"k": k}

        # KMeans
        km = KMeans(n_clusters=k, n_init=20, random_state=seed, max_iter=500)
        km_labels = km.fit_predict(X_pca)
        kmeans_models[k] = km
        row["km_inertia"] = km.inertia_
        row["km_silhouette"] = silhouette_score(X_pca, km_labels)
        row["km_calinski"] = calinski_harabasz_score(X_pca, km_labels)
        row["km_davies"] = davies_bouldin_score(X_pca, km_labels)
        row["km_labels"] = km_labels

        # Spectral
        try:
            sc = SpectralClustering(
                n_clusters=k, affinity="rbf", random_state=seed, n_init=10
            )
            sc_labels = sc.fit_predict(X_pca)
            row["sc_silhouette"] = silhouette_score(X_pca, sc_labels)
            row["sc_labels"] = sc_labels
        except Exception as e:
            warnings.warn(f"Spectral k={k} failed: {e}")
            row["sc_silhouette"] = np.nan
            row["sc_labels"] = None

        # Agglomerative (Ward)
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        ag_labels = ag.fit_predict(X_pca)
        row["ag_silhouette"] = silhouette_score(X_pca, ag_labels)
        row["ag_labels"] = ag_labels

        print(f"  k={k}: KM sil={row['km_silhouette']:.4f}, "
              f"SC sil={row.get('sc_silhouette', 'N/A'):.4f}, "
              f"AG sil={row['ag_silhouette']:.4f}")

        results.append(row)

    return results, kmeans_models


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------
def plot_pca_variance(pca, output_dir):
    """Plot 1: PCA explained variance curve."""
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum) + 1), cum, "b-o", markersize=2)
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Explained Variance")
    ax.axhline(0.95, color="r", ls="--", label="95% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_pca_variance.png"), dpi=150)
    plt.close(fig)


def plot_elbow(results, output_dir):
    """Plot 2: KMeans inertia (elbow) vs k."""
    ks = [r["k"] for r in results]
    inertias = [r["km_inertia"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, inertias, "bo-")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("KMeans Inertia")
    ax.set_title("Elbow Plot")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_elbow.png"), dpi=150)
    plt.close(fig)


def plot_silhouette_comparison(results, output_dir):
    """Plot 3: Silhouette score vs k for all 3 methods."""
    ks = [r["k"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, [r["km_silhouette"] for r in results], "bo-", label="KMeans")
    sc_sils = [r.get("sc_silhouette", np.nan) for r in results]
    ax.plot(ks, sc_sils, "rs-", label="Spectral")
    ax.plot(ks, [r["ag_silhouette"] for r in results], "g^-", label="Agglomerative")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score Comparison")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_silhouette_comparison.png"), dpi=150)
    plt.close(fig)


def plot_calinski_davies(results, output_dir):
    """Plot 4: Calinski-Harabasz and Davies-Bouldin vs k."""
    ks = [r["k"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ks, [r["km_calinski"] for r in results], "bo-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Calinski-Harabasz Index")
    ax1.set_title("Calinski-Harabasz (higher = better)")
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, [r["km_davies"] for r in results], "ro-")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Davies-Bouldin Index")
    ax2.set_title("Davies-Bouldin (lower = better)")
    ax2.set_xticks(ks)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_calinski_davies.png"), dpi=150)
    plt.close(fig)


def plot_silhouette_diagram(X_pca, labels, k, output_dir):
    """Plot 5: Silhouette diagram for optimal k."""
    sample_sil = silhouette_samples(X_pca, labels)
    avg_sil = silhouette_score(X_pca, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(k):
        ith_sil = sample_sil[labels == i]
        ith_sil.sort()
        size_i = len(ith_sil)
        y_upper = y_lower + size_i

        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith_sil,
            facecolor=color, edgecolor=color, alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color="red", ls="--", label=f"Mean sil = {avg_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Diagram (k={k})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_silhouette_diagram.png"), dpi=150)
    plt.close(fig)


def plot_tsne(X_pca, labels, adhd_labels, k, output_dir, seed=2025):
    """Plot 6: t-SNE coloured by cluster and by ADHD label."""
    print("[Plot] Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                           s=5, alpha=0.6)
    ax1.set_title(f"t-SNE: Cluster (k={k})")
    ax1.legend(*scatter1.legend_elements(), title="Cluster", loc="best", markerscale=3)

    adhd_colors = ["#4CAF50", "#F44336"]
    for val, color, label in [(0, adhd_colors[0], "Non-ADHD"), (1, adhd_colors[1], "ADHD")]:
        mask = adhd_labels == val
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=5, alpha=0.5, label=label)
    ax2.set_title("t-SNE: ADHD Label")
    ax2.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_tsne.png"), dpi=150)
    plt.close(fig)
    return X_2d


def plot_umap(X_pca, labels, adhd_labels, k, output_dir, seed=2025):
    """Plot 7: UMAP coloured by cluster and ADHD label."""
    try:
        import umap
    except ImportError:
        print("[Plot] UMAP not installed — skipping UMAP plot.")
        return None

    print("[Plot] Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(X_pca)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                           s=5, alpha=0.6)
    ax1.set_title(f"UMAP: Cluster (k={k})")
    ax1.legend(*scatter1.legend_elements(), title="Cluster", loc="best", markerscale=3)

    adhd_colors = ["#4CAF50", "#F44336"]
    for val, color, label in [(0, adhd_colors[0], "Non-ADHD"), (1, adhd_colors[1], "ADHD")]:
        mask = adhd_labels == val
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=5, alpha=0.5, label=label)
    ax2.set_title("UMAP: ADHD Label")
    ax2.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_umap.png"), dpi=150)
    plt.close(fig)
    return X_2d


def plot_cluster_composition(labels, adhd_labels, k, output_dir):
    """Plot 8: Stacked bar chart of ADHD vs non-ADHD per cluster."""
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = np.arange(k)
    adhd_counts = np.array([np.sum((labels == c) & (adhd_labels == 1)) for c in clusters])
    non_adhd_counts = np.array([np.sum((labels == c) & (adhd_labels == 0)) for c in clusters])
    totals = adhd_counts + non_adhd_counts

    ax.bar(clusters, non_adhd_counts / totals * 100, label="Non-ADHD", color="#4CAF50")
    ax.bar(clusters, adhd_counts / totals * 100, bottom=non_adhd_counts / totals * 100,
           label="ADHD", color="#F44336")

    for c in clusters:
        ax.text(c, 50, f"n={totals[c]}", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Cluster Composition (k={k})")
    ax.set_xticks(clusters)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_cluster_composition.png"), dpi=150)
    plt.close(fig)


def plot_mean_connectivity_heatmaps(pheno, data_root, labels, k, n_rois,
                                     feature_type, use_fisher_z, output_dir,
                                     tr=0.8, freq_band=(0.01, 0.1)):
    """Plot 9: Mean connectivity heatmap per cluster (FC or coherence)."""
    n_cols = min(k, 4)
    n_rows = (k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if k == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    sids = pheno["subjectkey"].values
    is_coherence = (feature_type == "coherence")
    fs = 1.0 / tr

    for c in range(k):
        row, col = divmod(c, n_cols)
        ax = axes[row, col]
        mask = labels == c
        cluster_sids = sids[mask]

        # Compute mean connectivity for this cluster
        mat_sum = np.zeros((n_rois, n_rois), dtype=np.float64)
        for sid in cluster_sids:
            fmri = _load_and_crop(data_root, sid)
            if is_coherence:
                mat = _coherence_matrix(fmri, fs, freq_band)
            else:
                mat = np.corrcoef(fmri.T)
                np.clip(mat, -1.0, 1.0, out=mat)
                if use_fisher_z:
                    np.clip(mat, -0.9999, 0.9999, out=mat)
                    mat = np.arctanh(mat)
            mat_sum += mat
        mean_mat = mat_sum / len(cluster_sids)

        if is_coherence:
            im = ax.imshow(mean_mat, cmap="viridis", vmin=0, vmax=0.5, aspect="equal")
        else:
            im = ax.imshow(mean_mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="equal")
        ax.set_title(f"Cluster {c} (n={mask.sum()})")
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(k, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    label = "Coherence" if is_coherence else "FC"
    fig.suptitle(f"Mean {label} Matrices per Cluster (k={k})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_mean_connectivity_heatmaps.png"), dpi=150)
    plt.close(fig)


def plot_dendrogram(X_pca, output_dir):
    """Plot 10: Dendrogram from hierarchical clustering."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    print("[Plot] Computing linkage for dendrogram...")
    # Subsample if too many subjects for readability
    n = X_pca.shape[0]
    if n > 2000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 2000, replace=False)
        X_sub = X_pca[idx]
        title_suffix = " (2000 random subjects)"
    else:
        X_sub = X_pca
        title_suffix = ""

    Z = linkage(X_sub, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax,
               leaf_rotation=90, leaf_font_size=8, color_threshold=0)
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward){title_suffix}")
    ax.set_xlabel("Cluster / Subject index")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_dendrogram.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Statistical analysis
# ---------------------------------------------------------------------------
def statistical_analysis(pheno, labels, optimal_k, output_dir):
    """Cross-tabulation and per-cluster demographic analysis."""
    lines = []

    adhd = pheno["ADHD_label"].values.astype(int)

    # --- Cross-tabulation: cluster x ADHD ---
    ct = pd.crosstab(labels, adhd, margins=True)
    ct.index.name = "Cluster"
    ct.columns = [f"ADHD={c}" if c != "All" else "All" for c in ct.columns]
    lines.append("=" * 60)
    lines.append("Cross-tabulation: Cluster x ADHD Label")
    lines.append("=" * 60)
    lines.append(ct.to_string())

    # Chi-squared test
    contingency = pd.crosstab(labels, adhd)
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    lines.append(f"\nChi-squared test: chi2={chi2:.4f}, p={p_val:.4e}, dof={dof}")
    if p_val < 0.05:
        lines.append("=> Significant association between clusters and ADHD label (p < 0.05)")
    else:
        lines.append("=> No significant association (p >= 0.05)")

    # --- Per-cluster demographics ---
    lines.append("\n" + "=" * 60)
    lines.append("Per-cluster Demographics")
    lines.append("=" * 60)

    for c in range(optimal_k):
        mask = labels == c
        sub = pheno[mask]
        lines.append(f"\n--- Cluster {c} (n={mask.sum()}) ---")
        lines.append(f"  ADHD prevalence: {adhd[mask].mean():.3f}")

        if "age" in sub.columns:
            lines.append(f"  Age: mean={sub['age'].mean():.2f}, std={sub['age'].std():.2f}")
        if "sex" in sub.columns:
            sex_dist = sub["sex"].value_counts(normalize=True)
            lines.append(f"  Sex distribution:\n{sex_dist.to_string()}")
        if "abcd_site" in sub.columns:
            top_sites = sub["abcd_site"].value_counts().head(5)
            lines.append(f"  Top 5 sites:\n{top_sites.to_string()}")
        if "race.ethnicity" in sub.columns:
            race_dist = sub["race.ethnicity"].value_counts(normalize=True)
            lines.append(f"  Race/ethnicity:\n{race_dist.to_string()}")

    return "\n".join(lines)


def recommend_optimal_k(results):
    """Recommend optimal k based on combined metrics."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("Optimal k Recommendation")
    lines.append("=" * 60)

    # Rank each k across metrics (higher silhouette = better, lower DB = better,
    # higher CH = better)
    ks = [r["k"] for r in results]
    sil = np.array([r["km_silhouette"] for r in results])
    ch = np.array([r["km_calinski"] for r in results])
    db = np.array([r["km_davies"] for r in results])

    # Normalise to [0, 1] for each metric
    def _norm(arr, higher_better=True):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        normed = (arr - mn) / (mx - mn)
        return normed if higher_better else 1.0 - normed

    scores = _norm(sil) + _norm(ch) + _norm(db, higher_better=False)
    best_idx = int(np.argmax(scores))
    optimal_k = ks[best_idx]

    for i, k in enumerate(ks):
        marker = " <-- BEST" if i == best_idx else ""
        lines.append(
            f"  k={k}: sil={sil[i]:.4f}, CH={ch[i]:.1f}, DB={db[i]:.4f}, "
            f"composite={scores[i]:.3f}{marker}"
        )

    lines.append(f"\nRecommended optimal k = {optimal_k}")
    lines.append(
        "Reasoning: Combined ranking across silhouette (higher=better), "
        "Calinski-Harabasz (higher=better), and Davies-Bouldin (lower=better)."
    )
    return optimal_k, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ABCD fMRI FC Clustering Analysis")
    parser.add_argument("--data-root", type=str,
                        default="/pscratch/sd/j/junghoon/ABCD",
                        help="Root directory containing ABCD data")
    parser.add_argument("--output-dir", type=str,
                        default="./results/fc_clustering",
                        help="Output directory for plots and results")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-k", type=int, default=8,
                        help="Maximum number of clusters to try")
    parser.add_argument("--n-pca-components", type=int, default=0,
                        help="Override PCA components (0 = auto via variance threshold)")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                        help="Cumulative variance threshold for auto PCA")
    parser.add_argument("--use-fisher-z", action="store_true",
                        help="Apply Fisher z-transform (arctanh) to correlations/coherence")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP plot (if umap-learn not installed)")
    parser.add_argument("--feature-type", type=str, default="fc",
                        choices=["fc", "coherence"],
                        help="Connectivity feature type: 'fc' (Pearson) or 'coherence'")
    parser.add_argument("--tr", type=float, default=0.8,
                        help="fMRI repetition time in seconds (for coherence)")
    parser.add_argument("--freq-band", type=float, nargs=2, default=[0.01, 0.1],
                        metavar=("F_LOW", "F_HIGH"),
                        help="Frequency band for coherence averaging (Hz)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    feat_label = "Coherence" if args.feature_type == "coherence" else "FC"
    print("=" * 60)
    print(f"ABCD fMRI {feat_label} Clustering Analysis")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Feature type: {args.feature_type}")
    print(f"Seed: {args.seed}, Max k: {args.max_k}")
    print(f"Fisher z-transform: {args.use_fisher_z}")
    if args.feature_type == "coherence":
        print(f"TR: {args.tr}s, Freq band: {args.freq_band} Hz")
    print()

    # 1. Load subjects
    pheno = load_subjects(args.data_root)

    # 2. Compute connectivity features
    if args.feature_type == "coherence":
        fc_features, n_rois = compute_coherence_features(
            pheno, args.data_root,
            tr=args.tr, freq_band=tuple(args.freq_band),
            use_fisher_z=args.use_fisher_z,
        )
    else:
        fc_features, n_rois = compute_fc_features(
            pheno, args.data_root, args.use_fisher_z
        )

    # 3. PCA
    n_comp = args.n_pca_components if args.n_pca_components > 0 else None
    X_pca, pca, scaler = run_pca(fc_features, args.variance_threshold, n_comp)

    # Save PCA features for downstream reuse
    np.save(os.path.join(args.output_dir, "fc_pca_features.npy"), X_pca)

    # 4. Clustering
    print(f"\n[Clustering] Running KMeans, Spectral, Agglomerative for k=2..{args.max_k}")
    results, kmeans_models = run_clustering(X_pca, args.max_k, args.seed)

    # Determine optimal k
    optimal_k, rec_text = recommend_optimal_k(results)
    print(rec_text)

    # Get labels for the optimal k
    optimal_result = [r for r in results if r["k"] == optimal_k][0]
    optimal_labels = optimal_result["km_labels"]

    # 5. Visualisation
    print("\n[Plots] Generating visualisations...")
    plot_pca_variance(pca, args.output_dir)
    plot_elbow(results, args.output_dir)
    plot_silhouette_comparison(results, args.output_dir)
    plot_calinski_davies(results, args.output_dir)
    plot_silhouette_diagram(X_pca, optimal_labels, optimal_k, args.output_dir)
    plot_tsne(X_pca, optimal_labels, pheno["ADHD_label"].values, optimal_k,
              args.output_dir, args.seed)

    if not args.skip_umap:
        plot_umap(X_pca, optimal_labels, pheno["ADHD_label"].values, optimal_k,
                  args.output_dir, args.seed)

    plot_cluster_composition(optimal_labels, pheno["ADHD_label"].values, optimal_k,
                             args.output_dir)
    plot_mean_connectivity_heatmaps(
        pheno, args.data_root, optimal_labels, optimal_k, n_rois,
        args.feature_type, args.use_fisher_z, args.output_dir,
        tr=args.tr, freq_band=tuple(args.freq_band),
    )
    plot_dendrogram(X_pca, args.output_dir)

    # 6. Statistical analysis
    print("\n[Stats] Running statistical analysis...")
    stats_text = statistical_analysis(pheno, optimal_labels, optimal_k, args.output_dir)
    print(stats_text)

    # 7. Save outputs
    # Clustering metrics CSV
    metrics_rows = []
    for r in results:
        metrics_rows.append({
            "k": r["k"],
            "km_inertia": r["km_inertia"],
            "km_silhouette": r["km_silhouette"],
            "km_calinski": r["km_calinski"],
            "km_davies": r["km_davies"],
            "sc_silhouette": r.get("sc_silhouette", np.nan),
            "ag_silhouette": r["ag_silhouette"],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.output_dir, "clustering_metrics.csv"), index=False)

    # Cluster assignments CSV
    assign_df = pheno.copy()
    assign_df["km_cluster"] = optimal_labels
    # Also add labels for all k values
    for r in results:
        assign_df[f"km_cluster_k{r['k']}"] = r["km_labels"]
        if r.get("ag_labels") is not None:
            assign_df[f"ag_cluster_k{r['k']}"] = r["ag_labels"]
    assign_df.to_csv(os.path.join(args.output_dir, "cluster_assignments.csv"), index=False)

    # Summary text
    summary_lines = []
    summary_lines.append(f"ABCD fMRI {feat_label} Clustering — Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Feature type: {args.feature_type}")
    summary_lines.append(f"N subjects: {len(pheno)}")
    summary_lines.append(f"N ROIs: {n_rois}")
    summary_lines.append(f"N features (upper tri): {fc_features.shape[1]}")
    summary_lines.append(f"Fisher z-transform: {args.use_fisher_z}")
    if args.feature_type == "coherence":
        summary_lines.append(f"TR: {args.tr}s, Freq band: {args.freq_band} Hz")
    summary_lines.append(f"PCA components: {X_pca.shape[1]} "
                         f"(variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.4f})")
    summary_lines.append(f"Optimal k (recommended): {optimal_k}")
    summary_lines.append("")
    summary_lines.append(rec_text)
    summary_lines.append("")
    summary_lines.append(stats_text)

    summary_path = os.path.join(args.output_dir, "clustering_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n[Done] Summary saved to {summary_path}")
    print(f"[Done] All outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
