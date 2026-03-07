# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_2 | **K**: 3 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.4920

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 149 | External | 0.493±0.014 | 0.507±0.014 |
| 1 | 5 | Internal | 0.630±0.050 | 0.370±0.050 |
| 2 | 134 | Internal | 0.516±0.011 | 0.484±0.011 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 286.815 | 0.0000 | Yes |
| External | 286.815 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3390

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 88 | Internal | 3.465 | 1.358 |
| 1 | 107 | Internal | 1.929 | 1.261 |
| 2 | 93 | Internal | 4.153 | 1.436 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2398

### Cluster 0 — Top 10 ROIs by Importance (N=24)

| Rank | ROI | Network | Importance | Signed | Direction |
|:----:|:---:|---------|:----------:|:------:|:---------:|
| 1 | 171 | DefaultA | 1.3638 | +0.0370 | +ADHD |
| 2 | 92 | ContA | 1.3183 | +0.0032 | +ADHD |
| 3 | 163 | ContA | 1.2431 | +0.0051 | +ADHD |
| 4 | 73 | SalVentAttnB | 1.1945 | -0.0005 | -ADHD |
| 5 | 91 | ContA | 1.1274 | +0.0093 | +ADHD |
| 6 | 134 | DefaultC | 1.1072 | +0.0202 | +ADHD |
| 7 | 25 | SomMotA | 1.0736 | +0.0006 | +ADHD |
| 8 | 42 | DorsAttnA | 1.0608 | -0.0004 | -ADHD |
| 9 | 51 | DorsAttnB | 1.0484 | +0.0005 | +ADHD |
| 10 | 164 | ContA | 1.0444 | +0.0027 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 171 | DefaultA | +0.0370 |
| 121 | DefaultA | +0.0224 |
| 134 | DefaultC | +0.0202 |
| 130 | DefaultB | +0.0163 |
| 117 | DefaultA | +0.0114 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 102 | ContC | -0.0013 |
| 123 | DefaultB | -0.0008 |
| 172 | DefaultB | -0.0007 |
| 69 | SalVentAttnA | -0.0007 |
| 84 | Limbic_OFC | -0.0006 |

### Cluster 1 — Top 10 ROIs by Importance (N=95)

| Rank | ROI | Network | Importance | Signed | Direction |
|:----:|:---:|---------|:----------:|:------:|:---------:|
| 1 | 73 | SalVentAttnB | 0.9073 | -0.0007 | -ADHD |
| 2 | 171 | DefaultA | 0.8607 | +0.0006 | +ADHD |
| 3 | 134 | DefaultC | 0.8045 | -0.0015 | -ADHD |
| 4 | 25 | SomMotA | 0.7159 | -0.0015 | -ADHD |
| 5 | 80 | Limbic_TempPole | 0.6932 | -0.0008 | -ADHD |
| 6 | 42 | DorsAttnA | 0.6733 | -0.0012 | -ADHD |
| 7 | 96 | ContB | 0.6686 | -0.0015 | -ADHD |
| 8 | 92 | ContA | 0.6651 | +0.0013 | +ADHD |
| 9 | 83 | Limbic_OFC | 0.6650 | -0.0011 | -ADHD |
| 10 | 168 | DefaultA | 0.6642 | -0.0007 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 92 | ContA | +0.0013 |
| 91 | ContA | +0.0007 |
| 171 | DefaultA | +0.0006 |
| 87 | ContA | +0.0005 |
| 89 | ContA | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 138 | DefaultC | -0.0016 |
| 8 | VisPeri | -0.0015 |
| 25 | SomMotA | -0.0015 |
| 129 | DefaultB | -0.0015 |
| 96 | ContB | -0.0015 |

### Cluster 2 — Top 10 ROIs by Importance (N=169)

| Rank | ROI | Network | Importance | Signed | Direction |
|:----:|:---:|---------|:----------:|:------:|:---------:|
| 1 | 73 | SalVentAttnB | 0.6652 | -0.0001 | -ADHD |
| 2 | 171 | DefaultA | 0.6379 | +0.0004 | +ADHD |
| 3 | 134 | DefaultC | 0.6103 | -0.0000 | -ADHD |
| 4 | 18 | SomMotA | 0.5383 | -0.0005 | -ADHD |
| 5 | 80 | Limbic_TempPole | 0.5054 | -0.0000 | -ADHD |
| 6 | 168 | DefaultA | 0.4868 | +0.0004 | +ADHD |
| 7 | 3 | VisCent | 0.4856 | -0.0000 | -ADHD |
| 8 | 65 | SalVentAttnA | 0.4827 | +0.0001 | +ADHD |
| 9 | 82 | Limbic_OFC | 0.4776 | -0.0001 | -ADHD |
| 10 | 83 | Limbic_OFC | 0.4762 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 177 | DefaultC | +0.0006 |
| 120 | DefaultA | +0.0006 |
| 119 | DefaultA | +0.0005 |
| 100 | ContB | +0.0005 |
| 91 | ContA | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 144 | TempPar | -0.0005 |
| 138 | DefaultC | -0.0005 |
| 18 | SomMotA | -0.0005 |
| 135 | DefaultC | -0.0005 |
| 28 | SomMotA | -0.0005 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | 0.8875 | 0.5135 | 0.3579 |
| ContB | 0.7534 | 0.5051 | 0.3532 |
| ContC | 0.6583 | 0.4681 | 0.3351 |
| DefaultA | 0.7787 | 0.5224 | 0.3837 |
| DefaultB | 0.7005 | 0.5136 | 0.3917 |
| DefaultC | 0.8061 | 0.5589 | 0.4326 |
| DorsAttnA | 0.7363 | 0.4963 | 0.3595 |
| DorsAttnB | 0.7763 | 0.5062 | 0.3550 |
| Limbic_OFC | 0.7148 | 0.5752 | 0.4170 |
| Limbic_TempPole | 0.7252 | 0.5592 | 0.4031 |
| SalVentAttnA | 0.7939 | 0.5272 | 0.3729 |
| SalVentAttnB | 0.7548 | 0.5574 | 0.4273 |
| SomMotA | 0.7233 | 0.5078 | 0.3960 |
| SomMotB | 0.7957 | 0.4978 | 0.3447 |
| TempPar | 0.7629 | 0.5002 | 0.3717 |
| VisCent | 0.6839 | 0.4893 | 0.3945 |
| VisPeri | 0.7408 | 0.5171 | 0.3930 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | +0.0032 (+ADHD) | +0.0000 (+ADHD) | +0.0000 (+ADHD) |
| ContB | +0.0006 (+ADHD) | -0.0005 (-ADHD) | +0.0002 (+ADHD) |
| ContC | -0.0002 (-ADHD) | -0.0005 (-ADHD) | +0.0001 (+ADHD) |
| DefaultA | +0.0050 (+ADHD) | -0.0004 (-ADHD) | +0.0002 (+ADHD) |
| DefaultB | +0.0026 (+ADHD) | -0.0006 (-ADHD) | -0.0000 (-ADHD) |
| DefaultC | +0.0040 (+ADHD) | -0.0009 (-ADHD) | -0.0002 (-ADHD) |
| DorsAttnA | +0.0003 (+ADHD) | -0.0007 (-ADHD) | -0.0000 (-ADHD) |
| DorsAttnB | +0.0003 (+ADHD) | -0.0007 (-ADHD) | +0.0000 (+ADHD) |
| Limbic_OFC | -0.0003 (-ADHD) | -0.0009 (-ADHD) | -0.0001 (-ADHD) |
| Limbic_TempPole | -0.0004 (-ADHD) | -0.0006 (-ADHD) | +0.0001 (+ADHD) |
| SalVentAttnA | +0.0008 (+ADHD) | -0.0006 (-ADHD) | +0.0000 (+ADHD) |
| SalVentAttnB | +0.0007 (+ADHD) | -0.0007 (-ADHD) | +0.0001 (+ADHD) |
| SomMotA | +0.0008 (+ADHD) | -0.0008 (-ADHD) | -0.0001 (-ADHD) |
| SomMotB | +0.0006 (+ADHD) | -0.0010 (-ADHD) | +0.0000 (+ADHD) |
| TempPar | +0.0004 (+ADHD) | -0.0006 (-ADHD) | -0.0002 (-ADHD) |
| VisCent | +0.0015 (+ADHD) | -0.0006 (-ADHD) | -0.0001 (-ADHD) |
| VisPeri | +0.0006 (+ADHD) | -0.0007 (-ADHD) | -0.0001 (-ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
