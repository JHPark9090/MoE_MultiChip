# ABCD fMRI Connectivity Clustering Analysis

## Motivation

The Mixture-of-Experts (MoE) model for ABCD fMRI ADHD classification initially split experts by spatial ROI regions (arbitrary partitioning), which led to severe overfitting and test AUC of only 0.55. A more principled approach is **subject-subtype specialization**: each expert processes all ROIs but specializes in a neurobiological subtype of subjects.

This clustering analysis identifies subject heterogeneity in the ABCD fMRI dataset by grouping subjects based on their brain connectivity patterns. The number of discovered clusters becomes the number of experts in the redesigned MoE, and the cluster assignments inform the gating network.

**Data**: 4,550 valid subjects with HCP180-parcellated resting-state fMRI (180 ROIs, 363 timepoints) and binary ADHD labels (3,231 non-ADHD / 2,550 ADHD). Total fMRI on disk: 9,406 subjects across three parcellations (HCP180, HCP360, Schaefer).

> **Note (2026-02-15)**: The original dataset had only 5,472 of 9,406 fMRI files on disk due
> to incomplete data transfer. The missing 3,934 files were restored from
> `/global/cfs/cdirs/m4750/7.ROI/`, increasing the ADHD-labeled sample from 2,606 to 4,550.
> All preliminary results cited below are from the partial dataset and may differ from
> full-dataset results.

---

## Connectivity Feature Types

The script supports two complementary approaches to representing each subject's brain connectivity as a feature vector.

### Functional Connectivity (FC) Matrices

**What it computes**: For each subject, the Pearson correlation coefficient between every pair of ROI time series.

Given a subject's fMRI data as a matrix $X \in \mathbb{R}^{T \times R}$ (T=363 timepoints, R=180 ROIs), the FC matrix is:

$$\text{FC}_{ij} = \text{corr}(X_{:,i},\ X_{:,j})$$

This produces a symmetric 180x180 matrix with values in [-1, 1]. The diagonal is always 1 (self-correlation). We extract the upper triangle (excluding diagonal), yielding **16,110 features per subject**.

**Interpretation**: FC measures *temporal co-activation* between brain regions. High positive FC between two ROIs means they tend to increase and decrease activity together. Negative FC means anti-correlated activity (when one goes up, the other goes down). FC captures the spatial organization of brain networks (default mode, salience, fronto-parietal, etc.) without regard to the temporal frequency of those co-fluctuations.

**Fisher z-transform** (`--use-fisher-z`): Pearson correlations are bounded to [-1, 1] and have a skewed distribution near the boundaries, which violates assumptions of PCA (linear) and KMeans (Euclidean distance). The Fisher z-transform, $z = \text{arctanh}(r)$, maps correlations to an approximately normal, unbounded distribution. This is standard practice in neuroimaging and is recommended for this analysis.

**When to use FC**: FC is the default and most widely used representation in resting-state fMRI research. It directly captures network-level organization, which is the most interpretable basis for MoE expert specialization (e.g., one expert for subjects with strong default-mode connectivity, another for subjects with strong fronto-parietal connectivity).

### Coherence Matrices

**What it computes**: For each subject, the *magnitude-squared coherence* between every pair of ROI time series, averaged over a specific frequency band.

Coherence is the frequency-domain analogue of correlation. It measures how consistently two signals co-oscillate at each frequency. The implementation uses a Welch-like approach:

1. Segment the time series into overlapping windows with Hann tapering
2. Compute the FFT of each segment for all ROIs simultaneously
3. Accumulate the cross-spectral density (CSD) $S_{xy}(f)$ and auto-spectral densities $S_{xx}(f)$, $S_{yy}(f)$ across segments
4. Compute magnitude-squared coherence:

$$C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f) \cdot S_{yy}(f)}$$

5. Average over frequency bins within the specified band (default: 0.01-0.1 Hz)

This produces a symmetric 180x180 matrix with values in [0, 1]. We extract the upper triangle, yielding the same **16,110 features per subject**.

**Frequency considerations for ABCD fMRI**:
- TR = 0.8 seconds, so the sampling rate is 1.25 Hz
- Nyquist frequency = 0.625 Hz
- The resting-state fMRI signal of interest lies in the low-frequency fluctuation (LFF) band: **0.01-0.1 Hz**
- With 363 timepoints and segment length of 128, frequency resolution is approximately 0.0098 Hz
- There are roughly 10-12 frequency bins in the 0.01-0.1 Hz band

**Interpretation**: Coherence measures *frequency-specific coupling* between brain regions. Two ROIs can have low Pearson correlation (low FC) but high coherence at a specific frequency — meaning they co-oscillate at that frequency but with varying phase relationships or different behavior at other frequencies. Coherence is more sensitive to oscillatory coupling and less affected by slow drifts or non-stationarities.

**When to use coherence**: Coherence may reveal subtypes that FC misses, particularly if ADHD-related differences are frequency-specific (e.g., altered coupling in the 0.01-0.03 Hz vs 0.03-0.1 Hz sub-bands). It is also more robust to motion artifacts that affect specific frequency ranges.

### FC vs Coherence: Key Differences

| Property | FC (Pearson) | Coherence |
|----------|-------------|-----------|
| Domain | Time | Frequency |
| Value range | [-1, 1] | [0, 1] |
| Captures anti-correlations | Yes | No (magnitude-squared) |
| Frequency-specific | No (broadband) | Yes (band-selectable) |
| Sensitivity to slow drifts | High | Low (windowed) |
| Standard in neuroimaging | Very common | Less common but growing |
| Computational cost | Fast (single corrcoef) | Moderate (Welch FFT) |

---

## Dimensionality Reduction

### StandardScaler

Before PCA, all 16,110 features are standardized to zero mean and unit variance using `sklearn.preprocessing.StandardScaler`, fitted on the full dataset. This is necessary because different ROI pairs have different variance scales — without scaling, PCA would be dominated by high-variance pairs and ignore low-variance but potentially informative pairs.

### PCA (Principal Component Analysis)

PCA finds the orthogonal directions (principal components) of maximum variance in the 16,110-dimensional feature space and projects data onto the top components.

**Why PCA is needed**:
- 16,110 features for 4,550 subjects creates a high-dimensional problem (p ~ 3.5n)
- Many FC/coherence features are correlated (nearby ROI pairs share similar connectivity patterns)
- KMeans uses Euclidean distance, which becomes less meaningful in very high dimensions (curse of dimensionality)
- PCA concentrates signal into fewer dimensions, improving clustering quality and speed

**Component selection**: By default, components are selected to capture 95% of cumulative explained variance (`--variance-threshold=0.95`). With the full 4,550-subject dataset, expect more components than the preliminary runs (~780 for FC, ~1,023 for coherence on 2,606 subjects). This can be overridden with `--n-pca-components`.

**Output**: `fc_pca_features.npy` — the PCA-transformed features, saved for downstream reuse in MoE training.

**Plot 1 — PCA Explained Variance Curve**: Shows cumulative explained variance vs number of components. A steep initial rise followed by a long tail indicates that the data has strong low-dimensional structure. The red dashed line marks the 95% threshold.

---

## Clustering Methods

Three clustering methods are run for each value of k from 2 to `--max-k` (default 8). Using multiple methods provides a robustness check — if all methods agree on the optimal k, the result is more trustworthy.

### KMeans

**Algorithm**: Partitions subjects into k clusters by iteratively assigning each subject to the nearest centroid (Euclidean distance in PCA space) and updating centroids to be the mean of assigned subjects.

**Configuration**: `n_init=20` (20 random initializations, best result kept), `max_iter=500`.

**Why KMeans is primary**: It directly maps to MoE — each cluster centroid becomes an expert's "specialty", and the gating network routes subjects to the nearest expert. KMeans produces compact, spherical clusters in PCA space, which aligns well with the MoE routing assumption.

**Limitations**: Assumes clusters are roughly spherical and equal-sized. Can struggle with elongated or irregularly shaped clusters.

### Spectral Clustering

**Algorithm**: Constructs a similarity graph from the data (using RBF/Gaussian kernel), computes the graph Laplacian, and clusters subjects based on the eigenvectors of the Laplacian.

**Configuration**: `affinity="rbf"`, `n_init=10`.

**Why included**: Spectral clustering can discover non-convex cluster shapes that KMeans misses. If spectral clustering finds a very different optimal k or much higher silhouette scores than KMeans, it suggests the data has non-spherical structure that KMeans is forcing into incorrect partitions.

**Limitations**: More computationally expensive ($O(n^3)$ for eigen-decomposition). The RBF kernel bandwidth is auto-selected, which may not be optimal. Warnings about disconnected graphs are common with fMRI data and indicate some subjects are very dissimilar from the rest.

### Agglomerative (Hierarchical) Clustering

**Algorithm**: Starts with each subject as its own cluster and iteratively merges the two most similar clusters using Ward linkage (minimizes within-cluster variance at each merge).

**Configuration**: `linkage="ward"`.

**Why included**: Produces a dendrogram (Plot 10) showing the full hierarchical structure. This reveals whether the data has a clear tree-like organization (suggesting natural subtypes) or a flat structure (suggesting the choice of k is somewhat arbitrary). Ward linkage is compatible with KMeans — both minimize within-cluster variance.

**Limitations**: Greedy (merges cannot be undone), so early mistakes propagate.

---

## Evaluation Metrics

No single metric definitively determines the optimal k. We combine four metrics:

### Silhouette Score

Measures how similar each subject is to its own cluster vs the nearest neighboring cluster. Ranges from -1 (wrong cluster) to +1 (perfectly clustered). Average over all subjects gives the overall silhouette score.

- **> 0.7**: Strong structure
- **0.5 - 0.7**: Reasonable structure
- **0.25 - 0.5**: Weak structure (common for fMRI data)
- **< 0.25**: Little to no structure

For fMRI FC data, silhouette scores of 0.05-0.30 are typical. Values above 0.2 are considered good for this data type.

### Calinski-Harabasz Index

Ratio of between-cluster variance to within-cluster variance. **Higher is better**. No absolute scale — only meaningful for comparing different k values on the same dataset. Tends to favor larger k, so should not be used alone.

### Davies-Bouldin Index

Average similarity between each cluster and its most similar cluster. **Lower is better** (0 = perfect separation). Less biased toward large k than Calinski-Harabasz.

### Inertia (Elbow Method)

Sum of squared distances from each subject to its cluster centroid (KMeans objective). Always decreases with k. The "elbow" — where the rate of decrease sharply flattens — suggests the optimal k. Subjective to interpret.

### Composite Score

The script computes a composite score by min-max normalizing each metric to [0, 1] and summing: silhouette (higher=better) + Calinski-Harabasz (higher=better) + inverted Davies-Bouldin (lower=better). The k with the highest composite score is recommended.

---

## Visualization

### Plot 1: PCA Explained Variance Curve

Shows how many PCA components are needed to capture the data's variance. A steep initial rise means the data has strong low-rank structure. The red dashed line marks the 95% threshold.

### Plot 2: Elbow Plot (KMeans Inertia)

KMeans inertia vs k. Look for an "elbow" — a sharp bend where adding more clusters stops reducing inertia substantially. If no clear elbow exists, the data lacks strong discrete cluster structure.

### Plot 3: Silhouette Score Comparison

Silhouette score vs k for all three clustering methods (KMeans, Spectral, Agglomerative). Agreement between methods strengthens confidence in the optimal k. If methods disagree, it suggests the cluster structure is ambiguous.

### Plot 4: Calinski-Harabasz and Davies-Bouldin

Two-panel plot. Left: CH index (higher=better). Right: DB index (lower=better). These metrics may suggest different optimal k values — that is normal and reflects different aspects of cluster quality.

### Plot 5: Silhouette Diagram

Per-sample silhouette coefficients for the optimal k, grouped by cluster. Each cluster appears as a horizontal "blade". Well-structured clusters have uniformly positive silhouette values. Clusters with many negative-silhouette samples are poorly separated.

**How to read**: Wide blades = large clusters. Blades extending far to the right = well-separated. Blades with negative values = subjects misassigned. Uneven blade widths = imbalanced cluster sizes.

### Plot 6: t-SNE

t-Distributed Stochastic Neighbor Embedding — a nonlinear dimensionality reduction that preserves local neighborhood structure. Two panels: colored by cluster assignment (left) and by ADHD label (right).

**How to read**: t-SNE preserves local structure but distorts global distances. Tight, separated groups indicate genuine clusters. Overlapping colors between the two panels mean clusters don't simply replicate the ADHD label (which is desirable — we want neural subtypes, not a proxy for diagnosis).

**Caveats**: t-SNE is stochastic and sensitive to perplexity. Cluster sizes and inter-cluster distances in t-SNE plots are not meaningful. Do not interpret global geometry.

### Plot 7: UMAP

Uniform Manifold Approximation and Projection — similar to t-SNE but better preserves global structure and is faster. Same two-panel layout.

**How to read**: UMAP tends to produce more compact clusters and preserves relative cluster distances better than t-SNE. If UMAP shows clear groups but t-SNE does not (or vice versa), trust the clustering metrics more than either visualization.

**Note**: Requires `umap-learn` package. Skipped if not installed or if `--skip-umap` is set.

### Plot 8: Cluster Composition

Stacked bar chart showing the percentage of ADHD vs non-ADHD subjects in each cluster. Annotated with total cluster size.

**How to read**: If all clusters have similar ADHD prevalence (~45% in ABCD), the clusters capture variability orthogonal to diagnosis. If some clusters are enriched for ADHD, the clusters partially align with clinical phenotype — these are the most interesting for MoE specialization.

### Plot 9: Mean Connectivity Heatmaps

Mean FC or coherence matrix (180x180) for each cluster. Uses `RdBu_r` colormap for FC (red=positive, blue=negative) and `viridis` for coherence (yellow=high, purple=low).

**How to read**: Compare heatmaps across clusters to identify which brain network connections differentiate the subtypes. Strong differences in specific blocks (e.g., default mode ↔ fronto-parietal) suggest the clusters capture meaningful network-level variability.

### Plot 10: Dendrogram

Hierarchical clustering tree (Ward linkage) truncated to 30 leaves for readability. Subsampled to 2,000 subjects if the dataset is larger.

**How to read**: Long vertical lines (large merge distances) indicate well-separated groups. If the first few merges are at much larger distances than subsequent ones, the data has k natural clusters where k equals the number of long initial branches.

---

## Statistical Analysis

### Chi-Squared Test (Cluster x ADHD)

Tests whether cluster membership is independent of ADHD diagnosis. A significant result (p < 0.05) means clusters are associated with ADHD — some clusters have higher or lower ADHD prevalence than expected by chance.

**Interpretation for MoE**: Significant association is informative but not required. The goal is neural subtypes, not ADHD prediction. Clusters that are orthogonal to ADHD may still improve MoE performance by reducing within-expert heterogeneity.

### Per-Cluster Demographics

Reports age, sex, site, and race/ethnicity distribution per cluster. **Critical for detecting confounds**: if clusters strongly correlate with acquisition site or demographics rather than neural connectivity, the clustering may reflect scanner differences or population stratification rather than genuine neural subtypes.

**Site effects**: If certain clusters are dominated by specific sites, consider applying ComBat harmonization to the FC/coherence features before clustering.

---

## Running the Analysis

### Prerequisites

Activate the conda environment:

```bash
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
```

### FC-Based Clustering (Default)

```bash
python abcd_fc_clustering.py \
    --data-root=/pscratch/sd/j/junghoon/ABCD \
    --output-dir=./results/fc_clustering \
    --seed=2025 \
    --max-k=8 \
    --use-fisher-z
```

### Coherence-Based Clustering

```bash
python abcd_fc_clustering.py \
    --feature-type=coherence \
    --tr=0.8 \
    --freq-band 0.01 0.1 \
    --data-root=/pscratch/sd/j/junghoon/ABCD \
    --output-dir=./results/coherence_clustering \
    --seed=2025 \
    --max-k=8
```

### Quick Local Test

```bash
python abcd_fc_clustering.py \
    --output-dir=./results/fc_clustering_test \
    --max-k=4 \
    --skip-umap \
    --use-fisher-z
```

### SLURM Submission

```bash
sbatch scripts/abcd_fc_clustering.sh
```

The SLURM script runs FC with Fisher-z by default. To run coherence instead, edit the script and change the python command.

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `/pscratch/sd/j/junghoon/ABCD` | Root directory containing ABCD data |
| `--output-dir` | `./results/fc_clustering` | Output directory for plots and results |
| `--seed` | `2025` | Random seed for reproducibility |
| `--max-k` | `8` | Maximum number of clusters to try (range: 2..max-k) |
| `--n-pca-components` | `0` | Override PCA components (0 = auto via variance threshold) |
| `--variance-threshold` | `0.95` | Cumulative variance threshold for auto PCA |
| `--use-fisher-z` | `False` | Apply Fisher z-transform (arctanh) to correlations/coherence |
| `--skip-umap` | `False` | Skip UMAP plot |
| `--feature-type` | `fc` | Feature type: `fc` (Pearson correlation) or `coherence` |
| `--tr` | `0.8` | fMRI repetition time in seconds (for coherence) |
| `--freq-band` | `0.01 0.1` | Frequency band for coherence averaging in Hz |

---

## Output Files

All outputs are saved to `--output-dir`:

| File | Description |
|------|-------------|
| `01_pca_variance.png` | PCA explained variance curve |
| `02_elbow.png` | KMeans inertia elbow plot |
| `03_silhouette_comparison.png` | Silhouette score vs k (3 methods) |
| `04_calinski_davies.png` | CH and DB indices vs k |
| `05_silhouette_diagram.png` | Per-sample silhouette for optimal k |
| `06_tsne.png` | t-SNE colored by cluster and ADHD label |
| `07_umap.png` | UMAP colored by cluster and ADHD label (if not skipped) |
| `08_cluster_composition.png` | ADHD prevalence per cluster |
| `09_mean_connectivity_heatmaps.png` | Mean FC/coherence matrix per cluster |
| `10_dendrogram.png` | Hierarchical clustering dendrogram |
| `clustering_summary.txt` | Full text summary with metrics and statistics |
| `clustering_metrics.csv` | Per-k metrics (inertia, silhouette, CH, DB) for all methods |
| `cluster_assignments.csv` | Per-subject: subjectkey, ADHD label, cluster labels for all k values, demographics |
| `fc_pca_features.npy` | PCA-transformed features (N x n_components) for downstream reuse |

---

## Interpreting Results

### Step 1: Check the Optimal k

Open `clustering_summary.txt` and look at the composite score table. The recommended k is a starting point, not a final answer.

**Common patterns**:
- k=2 often wins on all metrics — this is the coarsest split and may reflect a global signal (e.g., overall connectivity strength) rather than meaningful subtypes.
- If k=2 and k=3 have similar composite scores, try k=3 — it may capture a more interesting substructure at a small cost in cluster separation.
- For MoE, k=3-5 is a practical sweet spot. Fewer than 3 experts may not capture enough heterogeneity; more than 5 adds complexity without proportional benefit.

### Step 2: Check for Confounds

In `clustering_summary.txt`, examine per-cluster demographics:

- **Site distribution**: If clusters are dominated by specific sites, the clustering may reflect scanner/protocol differences. Consider ComBat harmonization.
- **Age/Sex**: Small differences are expected. Large differences (e.g., one cluster is 60% male, another 40%) may indicate demographic confounds.
- **Race/Ethnicity**: Similarly, check for systematic differences.

### Step 3: Examine the Visualizations

1. Look at **Plot 5 (silhouette diagram)**: Are all clusters reasonably positive? Any cluster with many negative-silhouette subjects is poorly defined.
2. Look at **Plot 6/7 (t-SNE/UMAP)**: Do visual clusters align with KMeans assignments? Is there visible separation?
3. Look at **Plot 8 (composition)**: Do clusters differ in ADHD prevalence? Modest differences (5-10%) are meaningful in a large sample.
4. Look at **Plot 9 (heatmaps)**: Can you see different network patterns across clusters?

### Step 4: Compare FC vs Coherence

Run both feature types and compare:

- **Same optimal k**: Both methods agree on the number of subtypes — strong evidence.
- **Different optimal k**: The methods capture different variability. Consider using the one with higher silhouette, or use both as input features.
- **Higher silhouette for coherence**: Frequency-domain coupling may be more informative for this dataset.
- **Stronger ADHD association for FC**: FC clusters may be more clinically relevant.

### Step 5: Use Results in MoE

1. Set `num_experts` = optimal k
2. Use `cluster_assignments.csv` to initialize or supervise the gating network
3. Use `fc_pca_features.npy` as auxiliary input to the gating network

---

## Expected Results for ABCD fMRI

### Full dataset (4,550 subjects — current)

- **Job ID**: 48949637 (FC clustering, submitted 2026-02-15)
- **Silhouette scores**: Expected 0.10-0.30 (typical for fMRI)
- **Optimal k**: Results pending; preliminary data suggested k=2 by metrics, but k=3-5 may be more useful for MoE
- **Runtime**: Estimated ~10-15 minutes for FC on 4,550 subjects (k=2..8, 32 CPUs)
- Results will be saved to `results/fc_clustering/`

### Preliminary results (2,606 subjects — partial dataset, outdated)

The following were obtained on only 57% of the valid subjects and may not hold on the full dataset:

| Feature | k | KM Silhouette | CH | DB | ADHD chi2 p |
|---------|---|---------------|------|------|-------------|
| FC (Fisher-z) | 2 | 0.265 | 1004 | 1.50 | 0.022 |
| FC (Fisher-z) | 3 | 0.145 | 685  | 2.00 | — |
| Coherence | 2 | 0.298 | 1278 | 1.37 | 0.052 |
| Coherence | 3 | 0.178 | 870  | 1.91 | — |

- PCA components (95% variance): ~780 for FC, ~1,023 for coherence
- Coherence produced better-separated clusters (higher silhouette) but weaker ADHD association
- Both methods recommended k=2 by composite score

### General expectations

- **Silhouette scores**: 0.10-0.30 is typical for fMRI — the data is inherently noisy and high-dimensional
- **Optimal k**: k=2 often wins on pure metrics; consider k=3-5 for MoE utility
- **Cluster-ADHD association**: Modest (p ~ 0.01-0.10). Clusters capture neural variability that is partially but not fully aligned with ADHD diagnosis
- **Site effects**: Some variation in site distribution across clusters is expected; strong site-cluster correlation warrants ComBat harmonization
- **Larger sample may improve clustering**: With 75% more subjects, clusters should be more stable and potentially reveal finer substructure (k=3-4 may become viable)
