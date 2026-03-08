# Heterogeneity Visualization Guide

**Date**: 2026-03-08 (updated with v5 results and multi-K analysis)
**Script**: `visualize_heterogeneity.py`
**Data source**: `analyze_heterogeneity.py` output (NPZ + JSON per model)
**Mapping**: Corrected Yeo-17 (v5, bilateral volumetric overlap)

## Overview

The Circuit MoE heterogeneity analysis asks: **are all ADHD+ subjects the same, or do they break into neurobiologically distinct subtypes?** Rather than treating ADHD as a single category, we cluster ADHD+ subjects based on how the trained model processes their brain data. The MoE architecture provides a natural hierarchy for this analysis — we can cluster at the level of circuit routing (gate weights), network-level processing (expert outputs), or fine-grained ROI importance (projection activations).

Five visualizations are generated per model, organized from coarsest (circuit) to finest (ROI) granularity, plus two summary metrics (silhouette sweep and cross-level concordance).

**Output directories**: `analysis/heterogeneity_v5_{model_type}_{config}/figures/`

| Model | K=3 Directory | Multi-K Directories (K=2,4,5) |
|-------|---------------|-------------------------------|
| Classical 4-expert | `heterogeneity_v5_classical_adhd_3/` | `..._k2/`, `..._k4/`, `..._k5/` |
| Classical 2-expert | `heterogeneity_v5_classical_adhd_2/` | `..._k2/`, `..._k4/`, `..._k5/` |
| Quantum 4-expert | `heterogeneity_v5_quantum_8q_d3_adhd_3/` | `..._k2/`, `..._k4/`, `..._k5/` |
| Quantum 2-expert | `heterogeneity_v5_quantum_8q_d3_adhd_2/` | `..._k2/`, `..._k4/`, `..._k5/` |

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
- **The gap between "Best K" and "Used K"** reflects a trade-off: K=2 may have the best separation but is too coarse to reveal subtypes; K=3 or higher allows exploration of finer structure at the cost of lower silhouette.

### v5 Scientific Findings

**Circuit-level (gate weight) silhouette**: Best at K=2 across all 4 models (0.39–0.97), confirming that routing is dominated by a binary split (near-uniform majority vs outlier minority). The load balancing auxiliary loss constrains gate weights, limiting circuit-level heterogeneity.

**ROI-level silhouette** (from the separate stability analysis, not this figure): Also best at K=2 (0.42–0.43) across all 4 models, but K=4-5 reveals additional subtypes (particularly SalVentAttnA) that K=2-3 misses. The trade-off between cluster quality (silhouette) and subtype resolution (number of meaningful subtypes) favors K=4-5 for comprehensive subtyping.

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

### v5 Scientific Findings

**4-expert models** (both classical and quantum):
- One dominant cluster (~94%) has near-uniform gate weights — load balancing prevents differentiation.
- A small rare cluster (1-5%) shows elevated routing to one specific expert. In the classical model, this is Salience-dominant; in the quantum model, it differs (SensoriMotor or DMN-dominant).
- **Key insight**: Circuit-level routing subtypes are architecture-dependent — classical and quantum models identify different rare routing patterns. However, they converge at ROI level (see Figure 4).

**2-expert models**:
- More balanced cluster sizes (~40/60%) because the 2D gate weight space allows cleaner separation.
- The quantum 2-expert model produces the most meaningful circuit-level split (statistically significant class-conditional routing, p=0.038).

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
- **Variance explained (axis labels)**: If PC1 explains >50%, a single dominant axis of variation drives the clustering. If PC1 and PC2 are roughly equal, the variation is more multi-dimensional.
- **Outliers**: Isolated points far from cluster centers may represent atypical ADHD presentations.

#### Radar Chart
- **Large polygon area**: That cluster produces strong expert outputs overall (high activation).
- **Asymmetric polygon**: One or two experts dominate, suggesting those brain circuits are most discriminative for that cluster's subjects.
- **Similar polygon shapes across clusters**: Clusters differ in magnitude (overall activation level) rather than pattern (which circuits are engaged).
- **Different polygon shapes**: Clusters represent qualitatively different patterns of brain circuit engagement — the strongest evidence for neurobiological subtypes.

### v5 Scientific Findings

- **Classical models** show higher network-level silhouette (0.35–0.40) than quantum models (0.17–0.21), meaning classical expert outputs are more separable into clusters.
- Quantum expert outputs are more distributed, possibly due to entanglement spreading information across output dimensions.
- Network-level clusters are model-specific and do not directly correspond to the ROI-level Limbic subtypes.

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

1. Run K-means (K=3 default, or K=2/4/5 for multi-K analyses) on the 180D signed ROI score vectors of ADHD+ subjects.
2. Apply PCA to reduce to 2D.
3. Plot each subject colored by cluster.

#### Network Heatmap (Right Panel)

1. For each of the 17 Yeo networks, compute the mean signed ROI score across all ROIs belonging to that network, within each cluster:
   ```
   net_score[cluster_k, network_j] = mean(signed_roi_scores[subjects_in_k, ROIs_in_network_j])
   ```

2. Display as a heatmap with a diverging red-blue colormap:
   - **Red (+)**: Positive signed score → network activation is associated with ADHD+ in this cluster
   - **Blue (-)**: Negative signed score → network activation is associated with ADHD- in this cluster
   - **White (0)**: No directional association

### How to Interpret

#### PCA Scatter
- **Separated clusters in PCA space**: Subjects have genuinely different ROI-level brain signatures.
- **One tight cluster + scattered outliers**: Most subjects share a common ROI pattern; outliers represent rare subtypes.
- **Variance explained**: If PC1 captures a large fraction (>30%), a single dominant axis of variation drives the clustering.

#### Network Heatmap
- **Row with strong red in Limbic_TempPole**: That cluster shows the Limbic_TempPole subtype — temporal pole regions strongly associated with ADHD+. Look for PeEc, TGv, TGd, TF, EC as top ROIs.
- **Row with strong red in Limbic_OFC**: The Limbic_OFC subtype — orbitofrontal regions associated with ADHD+. Look for OFC, 13l, pOFC, 10pp as top ROIs.
- **Row with strong red in SalVentAttnA** (K≥4 only): The insular/salience subtype — anterior/posterior insula and frontal operculum associated with ADHD+. Look for PI, AAIC, FOP1/2, PoI1 as top ROIs.
- **Near-white row**: A diffuse cluster with no strong directional associations. These subjects may represent milder or more heterogeneous ADHD presentations.
- **Row with strong blue across most networks**: These subjects show widespread negative signed scores — a general hypoactivation pattern.
- **Columns with consistent color across rows**: That network behaves similarly in all subtypes.
- **Columns with different colors across rows**: That network distinguishes subtypes.

### v5 Scientific Findings

**K=3 (default visualization):**
- **Limbic_TempPole subtype (~8% of ADHD+)**: Small cluster with extreme positive signed scores in temporal pole networks. PeEc, TGv, TGd, TF, EC are always the top 5 +ADHD ROIs. Replicated across all 4 models. **Tier 1 confidence.**
- **Limbic_OFC subtype (~33% of ADHD+)**: Larger cluster with OFC-dominant positive signed scores. OFC, 13l, 10pp, pOFC are characteristic ROIs. Replicated across all 4 models. **Tier 1 confidence.**
- **Diffuse majority (~59%)**: Near-zero signed scores across all networks.

**K=4-5 (multi-K analysis):**
- All K=3 subtypes persist, plus a **new SalVentAttnA (insular/salience) subtype** emerges (~35% of ADHD+), characterized by anterior insula (AAIC), posterior insula (PI), and frontal operculum (FOP1/2) ROIs. Replicated across all 4 models at both K=4 and K=5. **Tier 1 confidence.**
- The Limbic_TempPole cluster shrinks to a tight core (~5 subjects at K=5) with intensifying signed scores, while the ROI identity remains identical.

See `docs/Interpretability_Heterogeneity_v5_Results.md` Section 11.6 for complete multi-K analysis.

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
- **One high bar (e.g., Network vs ROI = 0.5+)**: Those two levels partially agree — network-level clusters reflect similar structure as ROI-level clusters.
- **Circuit vs others near zero, Network vs ROI moderate**: Circuit-level routing is too constrained by load balancing to capture the same structure as network/ROI levels.

### v5 Scientific Findings

ARI values are near zero across all 4 models (typically 0.02–0.13), confirming that **heterogeneity is multi-dimensional**. The slightly higher Network-ROI concordance in some models suggests that expert output patterns and ROI importance scores share some structure — both capture aspects of how the model processes each subject's brain data.

**Practical implication**: Characterizing ADHD subtypes requires multi-level analysis. The Limbic_TempPole and Limbic_OFC subtypes exist only at Level 3 (ROI) — they are completely invisible at Level 1 (Circuit) or Level 2 (Network). Conversely, circuit-level routing differences capture complementary information about how the model processes different subjects.

---

## Summary: From Visualization to Scientific Claims

| Figure | Primary Question | Key Metric | v5 Strongest Finding |
|--------|-----------------|------------|---------------------|
| Silhouette Sweep | How many subtypes exist? | Silhouette score | K=2 optimal (0.42-0.97 depending on level), but K=4-5 reveals SalVentAttnA subtype |
| Circuit-Level Clusters | Do subtypes differ in brain circuit routing? | Gate weight profiles | Load balancing constrains routing; 94% in one cluster for 4-expert models |
| Network-Level Clusters | Do subtypes differ in expert processing? | Expert output norms (PCA + radar) | Classical more separable than quantum; model-specific patterns |
| ROI-Level Clusters | Do subtypes differ in individual brain region engagement? | Signed ROI scores (PCA + heatmap) | **Three robust subtypes**: Limbic_TempPole (~8%), Limbic_OFC (~33%), SalVentAttnA (~35% at K≥4) |
| Cross-Level ARI | Are subtypes consistent across levels? | Adjusted Rand Index | Near-zero ARI: independent heterogeneity dimensions |

### How the Visualizations Build a Narrative

1. **Silhouette sweep** establishes that natural clustering exists (high silhouette at K=2) but is limited at the circuit level.
2. **Circuit-level clusters** confirm this limitation — load balancing produces a single dominant routing pattern.
3. **Network-level clusters** reveal that looking deeper (at expert outputs rather than gate weights) uncovers model-specific subtypes with varying expert engagement patterns.
4. **ROI-level clusters** provide the finest resolution, discovering three robust subtypes replicated across all 4 models: Limbic_TempPole (memory/emotion), Limbic_OFC (reward/impulse control), and SalVentAttnA (attentional salience — visible at K≥4).
5. **Cross-level ARI** demonstrates that these three levels capture independent information, justifying the multi-level approach.

The overall narrative: **ADHD heterogeneity in the Circuit MoE model is real, multi-dimensional, and neurobiologically interpretable.** The MoE architecture provides a natural hierarchy for subtyping — from coarse routing patterns to fine-grained ROI signatures — and each level reveals structure invisible to the others. The most robust findings are the three ROI-level subtypes (Tier 1 confidence), which align with three major theoretical models of ADHD neurobiology: fronto-limbic (TempPole), fronto-striatal/OFC (reward), and salience/attention (insula).

### Multi-K Visualization Note

The default visualizations are generated at K=3. For multi-K results (K=2, 4, 5), additional output directories exist with the suffix `_k{K}` (e.g., `heterogeneity_v5_classical_adhd_3_k4/`). Each contains the same 5 figures but with the specified K value. Key differences:

- **K=2**: Cleanest separation (highest silhouette); shows the binary Limbic_TempPole vs OFC/mixed split.
- **K=4**: The SalVentAttnA subtype first emerges as a distinct cluster.
- **K=5**: Three clear subtypes visible (TempPole, OFC, SalVentAttnA) plus diffuse clusters. The Limbic_TempPole cluster shrinks to its extreme core (~5 subjects).

## Cross-References

- Full numerical results: `docs/Interpretability_Heterogeneity_v5_Results.md` (Sections 7-11)
- Multi-K analysis details: `docs/Interpretability_Heterogeneity_v5_Results.md` (Section 11.6)
- Subtype stability test: `test_subtype_stability.py`
- Interpretability visualization guide: `docs/Interpretability_Visualization_Guide.md`
- Confidence framework: `docs/Interpretability_Heterogeneity_v5_Results.md` (Section 12)
- Terminology (ADHD+ vs +ADHD): `docs/Interpretability_Heterogeneity_v5_Results.md` (Terminology section)

## References

1. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53-65.
2. Hubert, L., & Arabie, P. (1985). Comparing partitions. *J. Classification*, 2(1), 193-218. (Adjusted Rand Index)
3. Menon V, Uddin LQ. Saliency, switching, attention and control: a network model of insula function. *Brain Struct Funct*. 2010;214(5-6):655-667. doi:10.1007/s00429-010-0262-0
4. Cortese S, et al. Toward systems neuroscience of ADHD: a meta-analysis of 55 fMRI studies. *Am J Psychiatry*. 2012;169(10):1038-1055. doi:10.1176/appi.ajp.2012.11101521
5. Koirala S, et al. Neurobiology of attention-deficit hyperactivity disorder: historical challenges and emerging frontiers. *Nat Rev Neurosci*. 2024;25(12):759-775. doi:10.1038/s41583-024-00869-z
6. Pan N, et al. Mapping ADHD heterogeneity and biotypes by topological deviations in morphometric similarity networks. *JAMA Psychiatry*. 2026. doi:10.1001/jamapsychiatry.2026.0001
7. Hoogman M, et al. Brain imaging of the cortex in ADHD: a coordinated analysis of large-scale clinical and population-based samples. *Am J Psychiatry*. 2019;176(7):531-542. doi:10.1176/appi.ajp.2019.18091033
