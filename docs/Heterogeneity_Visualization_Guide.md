# Heterogeneity Visualization Guide

> **NOTE (2026-03-07):** The example visualizations referenced in this guide were generated using an **incorrect Yeo-17 network mapping**. The visualization pipeline itself is valid and will work correctly with v5 checkpoints (corrected mapping).

**Date**: 2026-03-07
**Script**: `visualize_heterogeneity.py`
**Data source**: `analyze_heterogeneity.py` output (NPZ + JSON per model)

## Overview

The Circuit MoE heterogeneity analysis asks: **are all ADHD+ subjects the same, or do they break into neurobiologically distinct subtypes?** Rather than treating ADHD as a single category, we cluster ADHD+ subjects based on how the trained model processes their brain data. The MoE architecture provides a natural hierarchy for this analysis — we can cluster at the level of circuit routing (gate weights), network-level processing (expert outputs), or fine-grained ROI importance (projection activations).

Five visualizations are generated per model, organized from coarsest (circuit) to finest (ROI) granularity, plus two summary metrics (silhouette sweep and cross-level concordance).

---

## Figure 1: Silhouette Sweep

**File**: `silhouette_sweep.pdf/png`

### What It Shows

A line plot of silhouette score versus the number of clusters K (from 2 to 6), computed on circuit-level gate weight vectors of ADHD+ subjects only.

### How It Is Calculated

1. Extract the gate weight vector for each ADHD+ test subject. For a 4-expert model, this is a 4-dimensional vector `[w_DMN, w_Executive, w_Salience, w_SensoriMotor]` where each weight represents how much the model routes that subject's data to each expert. These weights sum to 1 (softmax output).

2. For each K from 2 to 6, run K-means clustering on these vectors (scikit-learn `KMeans`, 10 random restarts).

3. Compute the silhouette score for each K. The silhouette score for a single sample is:

   ```
   s(i) = (b(i) - a(i)) / max(a(i), b(i))
   ```

   where `a(i)` is the mean distance from sample i to all other samples in the same cluster, and `b(i)` is the mean distance to all samples in the nearest neighboring cluster. The overall silhouette score is the mean of `s(i)` across all samples.

   - `s = +1`: sample is far from neighboring clusters (perfect clustering)
   - `s = 0`: sample is on the boundary between clusters
   - `s = -1`: sample is likely in the wrong cluster

4. Two vertical lines mark the optimal K (highest silhouette, red dashed) and the K actually used for detailed analysis (blue dotted, default K=3).

### How to Interpret

- **High silhouette at K=2 (e.g., 0.96)** means gate weight vectors naturally split into two well-separated groups. This is expected when load balancing keeps most subjects near uniform routing, with a small outlier group.
- **Declining silhouette as K increases** suggests the data does not naturally support many distinct clusters — additional clusters are artificially splitting a continuous distribution.
- **The gap between "Best K" and "Used K"** reflects a trade-off: K=2 may have the best separation but is too coarse to reveal subtypes; K=3 allows exploration of finer structure at the cost of lower silhouette.

### Scientific Findings

In the 4-expert classical model, K=2 achieves silhouette 0.96 and K=3 achieves 0.78. This indicates that circuit-level routing is dominated by a binary split (near-uniform vs outlier) rather than multiple distinct routing strategies. The load balancing auxiliary loss (alpha=0.1) constrains gate weights to stay near uniform, limiting the model's ability to produce diverse routing patterns — and thus limiting circuit-level heterogeneity.

---

## Figure 2: Circuit-Level Clusters

**File**: `circuit_level_clusters.pdf/png`

### What It Shows

Side-by-side bar charts, one per cluster, showing the mean gate weight for each expert circuit. Error bars show standard deviation within each cluster. A horizontal dashed line marks the uniform routing baseline (1/K_experts).

### How It Is Calculated

1. ADHD+ test subjects are clustered using K-means on their gate weight vectors (at the K chosen for detailed analysis, default K=3).

2. For each cluster, compute the mean and standard deviation of each expert's gate weight across all subjects in that cluster.

3. The "dominant circuit" label (shown in titles) is the expert with the highest mean gate weight in that cluster.

### How to Interpret

- **Bars near the dashed line (uniform baseline)**: The model routes this cluster's subjects roughly equally across all experts. No single brain circuit dominates.
- **Bars significantly above the baseline**: This cluster's subjects preferentially route through that expert. The model "decides" that this brain circuit is more informative for these subjects.
- **Imbalanced cluster sizes**: If one cluster contains 90%+ of subjects, circuit-level routing does not discriminate well between ADHD subtypes — most subjects are routed identically.
- **Small error bars**: Subjects within the cluster are homogeneous in their routing pattern.
- **Large error bars**: Within-cluster heterogeneity; the cluster may be artificially combining different routing patterns.

### Scientific Findings

In the 4-expert model with K=3:
- **Cluster 1 (N=272, 94.4%)** has near-uniform weights with slight Salience elevation (0.256). This is the "typical" ADHD routing pattern — the vast majority of subjects.
- **Cluster 0 (N=15, 5.2%)** shows moderate Salience elevation (0.289) with reduced Executive (0.227). These subjects may have a more salience-network-driven ADHD presentation.
- **Cluster 2 (N=1, 0.3%)** is a single extreme outlier with Salience weight 0.589 — the model routes nearly 60% of this subject's data through the Salience expert.

The imbalance reveals that **load-balanced gating suppresses circuit-level heterogeneity**. The auxiliary loss penalizes unequal expert utilization across the batch, which forces most subjects into similar routing patterns. Individual-level subtyping requires looking deeper — at the network or ROI level.

---

## Figure 3: Network-Level Clusters

**File**: `network_level_clusters.pdf/png`

### What It Shows

Two panels:
- **Left**: PCA scatter plot of ADHD+ subjects in expert-output space, colored by cluster.
- **Right**: Radar chart showing mean expert output norms per cluster.

### How It Is Calculated

#### PCA Scatter (Left Panel)

1. For each ADHD+ test subject, extract the expert output vectors. In a 4-expert model with hidden dim 64, each subject produces 4 vectors of dimension 64, one per expert.

2. Flatten to a single vector per subject: `(4 × 64) = 256` dimensions. This captures not just which expert is dominant (gate weights) but **what each expert computed** for this subject.

3. Run K-means (K=3) on these 256D vectors to produce cluster assignments.

4. Apply PCA to reduce the 256D vectors to 2D for visualization. The axis labels show the percentage of variance explained by each principal component.

5. Plot each subject as a point, colored by cluster assignment.

#### Radar Chart (Right Panel)

1. For each cluster, compute the mean L2 norm of each expert's output vector across subjects in that cluster:
   ```
   norm_k = mean(||expert_k_output||_2)  for subjects in cluster
   ```

2. Plot these norms as a polygon on a radar chart, one line per cluster. Larger values = that expert produced stronger (higher-magnitude) outputs for subjects in that cluster.

### How to Interpret

#### PCA Scatter
- **Well-separated colored groups**: Clusters capture genuine structure in expert output space. Subjects in different clusters have meaningfully different expert processing patterns.
- **Overlapping groups**: Clusters are less distinct; the 256D separation doesn't fully project into 2D.
- **Variance explained (axis labels)**: If PC1 explains >50%, a single dominant axis of variation drives the clustering. If PC1 and PC2 are roughly equal (e.g., 30%/25%), the variation is more multi-dimensional.
- **Outliers**: Isolated points far from cluster centers may represent atypical ADHD presentations.

#### Radar Chart
- **Large polygon area**: That cluster produces strong expert outputs overall (high activation).
- **Asymmetric polygon**: One or two experts dominate, suggesting those brain circuits are most discriminative for that cluster's subjects.
- **Similar polygon shapes across clusters**: Clusters differ in magnitude (overall activation level) rather than pattern (which circuits are engaged).
- **Different polygon shapes**: Clusters represent qualitatively different patterns of brain circuit engagement — the strongest evidence for neurobiological subtypes.

### Scientific Findings

In the 4-expert model:
- **Cluster 0 (N=71, Executive-dominant)**: Highest Executive expert norm (4.11), lowest SensoriMotor (1.80). On the radar chart, the polygon stretches toward Executive and Salience. These subjects may represent an ADHD subtype where cognitive control dysfunction is primary.
- **Cluster 1 (N=139, DMN-dominant)**: Highest DMN norm (4.58) and highest SensoriMotor (2.60). The polygon stretches toward DMN. This is the largest subgroup and may correspond to the "classic" ADHD presentation involving default mode network dysregulation.
- **Cluster 2 (N=78, Balanced)**: Relatively uniform expert norms. No single circuit dominates — these subjects may have a less differentiated ADHD phenotype.

The PCA scatter (PC1=52.2%, PC2=26.8%) shows three groups with partial overlap, indicating genuine but not perfectly separable subtypes. The 79% variance captured in 2D suggests the clustering structure is largely two-dimensional.

---

## Figure 4: ROI-Level Clusters

**File**: `roi_level_clusters.pdf/png`

### What It Shows

Two panels:
- **Left**: PCA scatter plot of ADHD+ subjects in signed-ROI-score space, colored by cluster.
- **Right**: Heatmap of mean signed ROI scores per cluster, aggregated to the Yeo 17-network level.

### How It Is Calculated

#### Signed ROI Scores

For each subject, a 180-dimensional "signed ROI importance" vector is computed:

```
signed_roi_score[i] = mean_activation[i] × projection_weight_norm[i]
```

where:
- `mean_activation[i]` = temporal mean of ROI i's fMRI signal for this subject (preserves sign — positive or negative mean activation)
- `projection_weight_norm[i]` = L2 norm of the input projection weights connecting ROI i to its expert's hidden representation (always positive — measures how much the model attends to this ROI)

The product is **signed**: a positive score means the ROI's activation pattern pushes toward the ADHD+ prediction; a negative score means it pushes toward ADHD-. This is distinct from absolute ROI importance (which ignores direction).

#### PCA Scatter (Left Panel)

1. Run K-means (K=3) on the 180D signed ROI score vectors of ADHD+ subjects.
2. Apply PCA to reduce to 2D.
3. Plot each subject colored by cluster.

#### Network Heatmap (Right Panel)

1. For each of the 17 Yeo networks, compute the mean signed ROI score across all ROIs belonging to that network, within each cluster:
   ```
   net_score[cluster_k, network_j] = mean(signed_roi_scores[subjects_in_k, ROIs_in_network_j])
   ```

2. Display as a heatmap with a diverging red-blue colormap:
   - **Red (+)**: Positive signed score → network activation is associated with ADHD+ in this cluster
   - **Blue (-)**: Negative signed score → network activation is associated with ADHD- (i.e., lower activation = more ADHD-like) in this cluster
   - **White (0)**: No directional association

### How to Interpret

#### PCA Scatter
- **Separated clusters in PCA space**: Subjects have genuinely different ROI-level brain signatures.
- **One tight cluster + scattered outliers**: Most subjects share a common ROI pattern; outliers represent rare subtypes.
- **Variance explained**: If PC1 captures a large fraction (>30%), a single dominant axis of variation (e.g., overall activation level) drives the clustering.

#### Network Heatmap
- **Uniformly blue row**: That cluster's subjects show negative signed scores across all networks — their brain activation is generally associated with ADHD- (typical hypoactivation pattern).
- **Uniformly red row**: That cluster shows positive scores across all networks — their activation is generally associated with ADHD+ (atypical hyperactivation pattern).
- **Mixed red/blue row**: That cluster has a specific pattern — some networks positively associated, others negatively. This is the most informative for subtyping.
- **Columns with consistent color across rows**: That network behaves similarly in all subtypes (e.g., SensoriMotor networks may be consistently positive across all clusters).
- **Columns with different colors across rows**: That network distinguishes subtypes (e.g., DefaultA is blue in Cluster 0 but red in Cluster 1 → DMN hypoactivation vs hyperactivation subtypes).

### Scientific Findings

In the 4-expert model:
- **Cluster 0 (N=92, 31.9%)**: All-blue heatmap. All networks show negative signed scores. This represents the "canonical" ADHD pattern — widespread hypoactivation, consistent with the population-level gradient saliency finding.
- **Cluster 1 (N=22, 7.6%)**: Strikingly different — strong red in DefaultA, DefaultB, DefaultC, ContA. This rare subtype shows **DMN hyperactivation** associated with ADHD+, the opposite of the population-level pattern. In the PCA scatter, these 22 subjects are clearly separated from the main group. This is the most scientifically interesting finding: a small but distinct ADHD subtype with reversed DMN polarity.
- **Cluster 2 (N=174, 60.4%)**: Near-white heatmap (very small scores). These subjects have a weak, diffuse ROI pattern — the model does not assign strong directional importance to any network. This may represent subjects near the ADHD/non-ADHD classification boundary.

The discovery of a DMN-positive subtype (Cluster 1) is notable because it challenges the uniform "DMN hypoactivation in ADHD" narrative from the literature (Castellanos & Proal 2012). This aligns with recent ADHD biotype work (Feng 2024, Pan 2026) showing that ADHD is neurobiologically heterogeneous, with distinct brain signatures mapping to different clinical presentations.

---

## Figure 5: Cross-Level Concordance

**File**: `cross_level_ari.pdf/png`

### What It Shows

A bar chart of Adjusted Rand Index (ARI) between all pairs of clustering levels: Circuit vs Network, Circuit vs ROI, and Network vs ROI.

### How It Is Calculated

The Adjusted Rand Index measures agreement between two clustering assignments on the same set of subjects:

1. For each pair of levels (e.g., Circuit and Network), both produce a cluster label for each ADHD+ subject.

2. ARI is computed as:
   ```
   ARI = (RI - Expected_RI) / (Max_RI - Expected_RI)
   ```
   where RI (Rand Index) counts the fraction of subject pairs that are either in the same cluster in both levels or in different clusters in both levels. The adjustment corrects for chance agreement.

   - `ARI = 1.0`: Perfect agreement — the two levels produce identical clusters.
   - `ARI = 0.0`: Agreement no better than random — the two levels capture unrelated structure.
   - `ARI < 0.0`: Agreement worse than random (anti-correlated clustering).

### How to Interpret

- **All bars near zero**: Each analysis level captures independent heterogeneity dimensions. A subject's circuit-level subtype does not predict their ROI-level subtype. This means the three levels provide complementary (not redundant) information about ADHD heterogeneity.
- **One high bar (e.g., Network vs ROI = 0.5+)**: Those two levels partially agree — network-level clusters reflect similar structure as ROI-level clusters, suggesting one level may be a coarser view of the other.
- **Circuit vs others near zero, Network vs ROI moderate**: Circuit-level routing (gate weights) is too constrained by load balancing to capture the same structure as network/ROI levels. The finer-grained levels share more structure.

### Scientific Findings

In the 4-expert model:
- Circuit vs Network: ARI = 0.026
- Circuit vs ROI: ARI = 0.067
- Network vs ROI: ARI = 0.131

All values are near zero, confirming that **heterogeneity is multi-dimensional**. The slightly higher Network-ROI concordance (0.131) suggests that expert output patterns and ROI importance scores share some structure — both capture aspects of how the model internally processes each subject's brain data, whereas circuit-level routing is too constrained to align with either.

This has a practical implication: **characterizing ADHD subtypes requires multi-level analysis**. Reporting only circuit-level (gate weight) clusters would miss the ROI-level DMN-positive subtype entirely. Reporting only ROI-level clusters would miss the network-level Executive-dominant vs DMN-dominant distinction. The MoE architecture's multi-level representations enable richer subtyping than a single analysis level could provide.

---

## Summary: From Visualization to Scientific Claims

| Figure | Primary Question | Key Metric | Strongest Finding |
|--------|-----------------|------------|-------------------|
| Silhouette Sweep | How many subtypes exist? | Silhouette score | K=2 optimal (0.96), suggesting binary split at circuit level |
| Circuit-Level Clusters | Do subtypes differ in brain circuit routing? | Gate weight profiles | Load balancing constrains routing; 94.4% in one cluster |
| Network-Level Clusters | Do subtypes differ in expert processing? | Expert output norms (PCA + radar) | 3 balanced subtypes: Executive-dominant, DMN-dominant, Balanced |
| ROI-Level Clusters | Do subtypes differ in individual brain region engagement? | Signed ROI scores (PCA + heatmap) | Rare DMN-positive subtype (7.6%) with reversed polarity |
| Cross-Level ARI | Are subtypes consistent across levels? | Adjusted Rand Index | Near-zero ARI: independent heterogeneity dimensions |

### How the Visualizations Build a Narrative

1. **Silhouette sweep** establishes that natural clustering exists (high silhouette at K=2) but is limited at the circuit level.
2. **Circuit-level clusters** confirm this limitation — load balancing produces a single dominant routing pattern.
3. **Network-level clusters** reveal that looking deeper (at expert outputs rather than gate weights) uncovers 3 balanced, interpretable subtypes.
4. **ROI-level clusters** provide the finest resolution, discovering a rare but distinctive DMN-positive subtype invisible at coarser levels.
5. **Cross-level ARI** demonstrates that these three levels capture independent information, justifying the multi-level approach.

The overall narrative: **ADHD heterogeneity in the Circuit MoE model is real but multi-dimensional.** The MoE architecture provides a natural hierarchy for subtyping — from coarse routing patterns to fine-grained ROI signatures — and each level reveals structure invisible to the others.

## References

1. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53-65.
2. Hubert, L., & Arabie, P. (1985). Comparing partitions. *J. Classification*, 2(1), 193-218. (Adjusted Rand Index)
3. Feng, A., et al. (2024). Functional imaging derived ADHD biotypes. *EClinicalMedicine*, 77, 102876.
4. Pan, N., et al. (2026). Mapping ADHD heterogeneity and biotypes. *JAMA Psychiatry*.
5. Castellanos, F. X., & Proal, E. (2012). Large-scale brain systems in ADHD. *Neuropsychopharmacology*, 37(1), 247-259.
