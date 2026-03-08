# ADHD Heterogeneity Analysis via Learned Representations

> **DEPRECATED (2026-03-07):** Generated with incorrect Yeo-17 mapping. ROI/network/circuit interpretations are not neurobiologically valid. Will be regenerated with corrected mapping (v5).

**Model**: classical | **Config**: adhd_3 | **K**: 3 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.7796

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 15 | Salience | 0.250±0.021 | 0.227±0.015 | 0.289±0.020 | 0.234±0.009 |
| 1 | 272 | Salience | 0.251±0.004 | 0.250±0.006 | 0.256±0.004 | 0.243±0.001 |
| 2 | 1 | Salience | 0.142±0.000 | 0.134±0.000 | 0.589±0.000 | 0.135±0.000 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 150.893 | 0.0000 | Yes |
| Executive | 249.749 | 0.0000 | Yes |
| Salience | 1679.090 | 0.0000 | Yes |
| SensoriMotor | 1034.233 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.2959

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 71 | Executive | 3.050 | 4.111 | 3.371 | 1.804 |
| 1 | 139 | DMN | 4.581 | 3.396 | 3.813 | 2.598 |
| 2 | 78 | DMN | 3.566 | 3.456 | 3.334 | 2.396 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2499

### Cluster 0 — Top 10 ROIs by Importance (N=92)

| Rank | ROI | Network | Importance | Signed | Direction |
|:----:|:---:|---------|:----------:|:------:|:---------:|
| 1 | 73 | SalVentAttnB | 1.1857 | -0.0009 | -ADHD |
| 2 | 171 | DefaultA | 1.0520 | +0.0007 | +ADHD |
| 3 | 71 | SalVentAttnA | 0.9098 | +0.0002 | +ADHD |
| 4 | 161 | SalVentAttnA | 0.8745 | -0.0020 | -ADHD |
| 5 | 25 | SomMotA | 0.8693 | -0.0018 | -ADHD |
| 6 | 83 | Limbic_OFC | 0.8690 | -0.0012 | -ADHD |
| 7 | 10 | VisPeri | 0.8672 | -0.0006 | -ADHD |
| 8 | 74 | SalVentAttnB | 0.8637 | -0.0009 | -ADHD |
| 9 | 29 | SomMotA | 0.8574 | -0.0018 | -ADHD |
| 10 | 134 | DefaultC | 0.8518 | -0.0020 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 92 | ContA | +0.0018 |
| 91 | ContA | +0.0009 |
| 89 | ContA | +0.0007 |
| 171 | DefaultA | +0.0007 |
| 87 | ContA | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 160 | SalVentAttnA | -0.0021 |
| 161 | SalVentAttnA | -0.0020 |
| 134 | DefaultC | -0.0020 |
| 35 | SomMotB | -0.0019 |
| 25 | SomMotA | -0.0018 |

### Cluster 1 — Top 10 ROIs by Importance (N=22)

| Rank | ROI | Network | Importance | Signed | Direction |
|:----:|:---:|---------|:----------:|:------:|:---------:|
| 1 | 171 | DefaultA | 1.7046 | +0.0489 | +ADHD |
| 2 | 73 | SalVentAttnB | 1.6216 | -0.0009 | -ADHD |
| 3 | 92 | ContA | 1.5625 | +0.0029 | +ADHD |
| 4 | 60 | SalVentAttnA | 1.5001 | +0.0061 | +ADHD |
| 5 | 71 | SalVentAttnA | 1.4081 | +0.0010 | +ADHD |
| 6 | 161 | SalVentAttnA | 1.3942 | +0.0019 | +ADHD |
| 7 | 163 | ContA | 1.3749 | +0.0056 | +ADHD |
| 8 | 29 | SomMotA | 1.3262 | +0.0006 | +ADHD |
| 9 | 164 | ContA | 1.3132 | +0.0030 | +ADHD |
| 10 | 25 | SomMotA | 1.2921 | +0.0007 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 171 | DefaultA | +0.0489 |
| 134 | DefaultC | +0.0236 |
| 121 | DefaultA | +0.0235 |
| 130 | DefaultB | +0.0176 |
| 117 | DefaultA | +0.0130 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 102 | ContC | -0.0022 |
| 69 | SalVentAttnA | -0.0015 |
| 114 | DefaultA | -0.0011 |
| 62 | SalVentAttnA | -0.0011 |
| 84 | Limbic_OFC | -0.0011 |

### Cluster 2 — Top 10 ROIs by Importance (N=174)

| Rank | ROI | Network | Importance | Signed | Direction |
|:----:|:---:|---------|:----------:|:------:|:---------:|
| 1 | 73 | SalVentAttnB | 0.8774 | -0.0001 | -ADHD |
| 2 | 171 | DefaultA | 0.7794 | +0.0005 | +ADHD |
| 3 | 74 | SalVentAttnB | 0.6788 | +0.0000 | +ADHD |
| 4 | 29 | SomMotA | 0.6553 | +0.0000 | +ADHD |
| 5 | 134 | DefaultC | 0.6524 | +0.0001 | +ADHD |
| 6 | 83 | Limbic_OFC | 0.6290 | -0.0003 | -ADHD |
| 7 | 159 | SalVentAttnA | 0.6199 | +0.0004 | +ADHD |
| 8 | 10 | VisPeri | 0.6192 | +0.0001 | +ADHD |
| 9 | 5 | VisCent | 0.6191 | -0.0002 | -ADHD |
| 10 | 150 | VisPeri | 0.6174 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 120 | DefaultA | +0.0007 |
| 177 | DefaultC | +0.0006 |
| 119 | DefaultA | +0.0006 |
| 171 | DefaultA | +0.0005 |
| 91 | ContA | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Network | Signed Score |
|:---:|---------|:------------:|
| 126 | DefaultB | -0.0006 |
| 22 | SomMotA | -0.0006 |
| 144 | TempPar | -0.0006 |
| 136 | DefaultC | -0.0006 |
| 18 | SomMotA | -0.0005 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | 0.6456 | 1.1116 | 0.4463 |
| ContB | 0.5845 | 0.8829 | 0.4096 |
| ContC | 0.6397 | 0.9066 | 0.4567 |
| DefaultA | 0.5947 | 0.9007 | 0.4369 |
| DefaultB | 0.5817 | 0.8172 | 0.4451 |
| DefaultC | 0.5834 | 0.8446 | 0.4517 |
| DorsAttnA | 0.6067 | 0.8778 | 0.4390 |
| DorsAttnB | 0.6061 | 0.9156 | 0.4220 |
| Limbic_OFC | 0.8044 | 1.0370 | 0.5869 |
| Limbic_TempPole | 0.7684 | 1.0265 | 0.5573 |
| SalVentAttnA | 0.7730 | 1.1700 | 0.5407 |
| SalVentAttnB | 0.8508 | 1.1863 | 0.6556 |
| SomMotA | 0.6542 | 0.9288 | 0.5074 |
| SomMotB | 0.6970 | 1.0824 | 0.4695 |
| TempPar | 0.5783 | 0.8712 | 0.4264 |
| VisCent | 0.6282 | 0.8715 | 0.5050 |
| VisPeri | 0.7175 | 1.0127 | 0.5409 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | +0.0001 (+ADHD) | +0.0037 (+ADHD) | +0.0000 (+ADHD) |
| ContB | -0.0004 (-ADHD) | +0.0007 (+ADHD) | +0.0002 (+ADHD) |
| ContC | -0.0007 (-ADHD) | -0.0005 (-ADHD) | +0.0001 (+ADHD) |
| DefaultA | -0.0004 (-ADHD) | +0.0059 (+ADHD) | +0.0002 (+ADHD) |
| DefaultB | -0.0007 (-ADHD) | +0.0032 (+ADHD) | -0.0001 (-ADHD) |
| DefaultC | -0.0009 (-ADHD) | +0.0045 (+ADHD) | -0.0002 (-ADHD) |
| DorsAttnA | -0.0007 (-ADHD) | +0.0005 (+ADHD) | -0.0001 (-ADHD) |
| DorsAttnB | -0.0008 (-ADHD) | +0.0005 (+ADHD) | -0.0000 (-ADHD) |
| Limbic_OFC | -0.0011 (-ADHD) | -0.0006 (-ADHD) | -0.0002 (-ADHD) |
| Limbic_TempPole | -0.0008 (-ADHD) | -0.0008 (-ADHD) | +0.0001 (+ADHD) |
| SalVentAttnA | -0.0008 (-ADHD) | +0.0011 (+ADHD) | +0.0000 (+ADHD) |
| SalVentAttnB | -0.0010 (-ADHD) | +0.0010 (+ADHD) | +0.0001 (+ADHD) |
| SomMotA | -0.0010 (-ADHD) | +0.0011 (+ADHD) | -0.0001 (-ADHD) |
| SomMotB | -0.0014 (-ADHD) | +0.0008 (+ADHD) | -0.0000 (-ADHD) |
| TempPar | -0.0006 (-ADHD) | +0.0004 (+ADHD) | -0.0002 (-ADHD) |
| VisCent | -0.0007 (-ADHD) | +0.0022 (+ADHD) | -0.0001 (-ADHD) |
| VisPeri | -0.0009 (-ADHD) | +0.0010 (+ADHD) | -0.0001 (-ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
