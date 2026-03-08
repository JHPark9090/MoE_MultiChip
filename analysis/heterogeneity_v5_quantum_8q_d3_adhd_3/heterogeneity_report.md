# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_3 | **K**: 3 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.4421

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 122 | Executive | 0.250±0.001 | 0.253±0.002 | 0.246±0.002 | 0.252±0.001 |
| 1 | 7 | Salience | 0.241±0.002 | 0.239±0.003 | 0.267±0.004 | 0.254±0.002 |
| 2 | 159 | SensoriMotor | 0.248±0.001 | 0.250±0.001 | 0.249±0.001 | 0.252±0.000 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 210.356 | 0.0000 | Yes |
| Executive | 291.165 | 0.0000 | Yes |
| Salience | 407.118 | 0.0000 | Yes |
| SensoriMotor | 41.875 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.1723

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 71 | Executive | 1.163 | 1.361 | 1.169 | 1.288 |
| 1 | 106 | Salience | 1.501 | 1.520 | 1.525 | 1.171 |
| 2 | 111 | Salience | 2.081 | 1.736 | 2.496 | 1.275 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2444

### Cluster 0 — Top 10 ROIs by Importance (N=93)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 70 | 9p | TempPar | 0.7968 | -0.0009 | -ADHD |
| 2 | 23 | A1 | SomMotB | 0.7567 | -0.0009 | -ADHD |
| 3 | 80 | IFSp | ContB | 0.7326 | -0.0009 | -ADHD |
| 4 | 134 | TF | Limbic_TempPole | 0.7318 | -0.0016 | -ADHD |
| 5 | 39 | 24dd | SomMotA | 0.7286 | -0.0010 | -ADHD |
| 6 | 92 | OFC | Limbic_OFC | 0.7167 | +0.0014 | +ADHD |
| 7 | 97 | s6-8 | DefaultC | 0.7143 | -0.0011 | -ADHD |
| 8 | 128 | STSdp | DefaultA | 0.7074 | -0.0013 | -ADHD |
| 9 | 43 | 6ma | SalVentAttnA | 0.6989 | -0.0012 | -ADHD |
| 10 | 149 | PGi | DefaultC | 0.6985 | -0.0011 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0014 |
| 91 | 13l | Limbic_OFC | +0.0006 |
| 89 | 10pp | Limbic_OFC | +0.0005 |
| 87 | 10v | Limbic_OFC | +0.0005 |
| 165 | pOFC | Limbic_OFC | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 34 | 31pv | DefaultC | -0.0019 |
| 129 | STSvp | TempPar | -0.0019 |
| 160 | 31pd | DefaultC | -0.0018 |
| 35 | 5m | SomMotA | -0.0017 |
| 41 | 7AL | DorsAttnB | -0.0017 |

### Cluster 1 — Top 10 ROIs by Importance (N=171)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5662 | +0.0002 | +ADHD |
| 2 | 134 | TF | Limbic_TempPole | 0.5566 | +0.0001 | +ADHD |
| 3 | 175 | STSva | TempPar | 0.5506 | -0.0001 | -ADHD |
| 4 | 149 | PGi | DefaultC | 0.5450 | +0.0000 | +ADHD |
| 5 | 159 | VMV2 | VisPeri | 0.5420 | +0.0003 | +ADHD |
| 6 | 4 | V3 | VisCent | 0.5409 | -0.0001 | -ADHD |
| 7 | 80 | IFSp | ContB | 0.5356 | -0.0000 | -ADHD |
| 8 | 128 | STSdp | DefaultA | 0.5303 | +0.0000 | +ADHD |
| 9 | 70 | 9p | TempPar | 0.5276 | +0.0003 | +ADHD |
| 10 | 126 | PHA3 | DefaultB | 0.5275 | -0.0004 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0009 |
| 119 | H | Limbic_TempPole | +0.0007 |
| 167 | Ig | SomMotB | +0.0006 |
| 120 | ProS | VisPeri | +0.0006 |
| 91 | 13l | Limbic_OFC | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 144 | IP1 | ContB | -0.0006 |
| 136 | PHT | ContB | -0.0006 |
| 143 | IP2 | ContB | -0.0005 |
| 122 | STGa | DefaultA | -0.0005 |
| 28 | 7Pm | ContA | -0.0005 |

### Cluster 2 — Top 10 ROIs by Importance (N=24)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.4176 | +0.0034 | +ADHD |
| 2 | 163 | 25 | Limbic_OFC | 1.3294 | +0.0054 | +ADHD |
| 3 | 165 | pOFC | Limbic_OFC | 1.2986 | +0.0080 | +ADHD |
| 4 | 164 | s32 | Limbic_OFC | 1.2967 | +0.0033 | +ADHD |
| 5 | 39 | 24dd | SomMotA | 1.2295 | +0.0013 | +ADHD |
| 6 | 177 | PI | SalVentAttnA | 1.2100 | +0.0089 | +ADHD |
| 7 | 60 | a24 | DefaultC | 1.1556 | +0.0045 | +ADHD |
| 8 | 91 | 13l | Limbic_OFC | 1.1322 | +0.0094 | +ADHD |
| 9 | 64 | 10r | DefaultC | 1.1229 | +0.0019 | +ADHD |
| 10 | 54 | 6mp | SomMotA | 1.1218 | +0.0001 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.0281 |
| 121 | PeEc | Limbic_TempPole | +0.0238 |
| 134 | TF | Limbic_TempPole | +0.0184 |
| 130 | TGd | Limbic_TempPole | +0.0173 |
| 117 | EC | Limbic_TempPole | +0.0127 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0019 |
| 123 | PBelt | SomMotB | -0.0011 |
| 172 | MBelt | SomMotB | -0.0009 |
| 69 | 8BL | TempPar | -0.0009 |
| 84 | a9-46v | ContC | -0.0008 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | 0.5775 | 0.4162 | 0.9305 |
| ContB | 0.6272 | 0.4595 | 0.8717 |
| ContC | 0.6079 | 0.4527 | 0.8439 |
| DefaultA | 0.6299 | 0.4701 | 0.8140 |
| DefaultB | 0.6216 | 0.4740 | 0.8207 |
| DefaultC | 0.6455 | 0.4678 | 1.0039 |
| DorsAttnA | 0.5695 | 0.4500 | 0.8259 |
| DorsAttnB | 0.6080 | 0.4283 | 0.9231 |
| Limbic_OFC | 0.6011 | 0.4025 | 1.1658 |
| Limbic_TempPole | 0.5924 | 0.4305 | 0.9173 |
| SalVentAttnA | 0.6098 | 0.4275 | 0.9211 |
| SalVentAttnB | 0.6339 | 0.4488 | 0.9174 |
| SomMotA | 0.6378 | 0.4092 | 1.0147 |
| SomMotB | 0.6289 | 0.4533 | 0.8817 |
| TempPar | 0.6228 | 0.4633 | 0.8716 |
| VisCent | 0.5496 | 0.4818 | 0.6984 |
| VisPeri | 0.5544 | 0.4687 | 0.7311 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | -0.0011 (-ADHD) | -0.0001 (-ADHD) | +0.0008 (+ADHD) |
| ContB | -0.0009 (-ADHD) | -0.0001 (-ADHD) | +0.0001 (+ADHD) |
| ContC | -0.0010 (-ADHD) | +0.0000 (+ADHD) | +0.0011 (+ADHD) |
| DefaultA | -0.0010 (-ADHD) | -0.0002 (-ADHD) | +0.0021 (+ADHD) |
| DefaultB | -0.0007 (-ADHD) | -0.0000 (-ADHD) | +0.0016 (+ADHD) |
| DefaultC | -0.0009 (-ADHD) | +0.0000 (+ADHD) | +0.0009 (+ADHD) |
| DorsAttnA | -0.0006 (-ADHD) | -0.0003 (-ADHD) | +0.0014 (+ADHD) |
| DorsAttnB | -0.0012 (-ADHD) | -0.0000 (-ADHD) | +0.0001 (+ADHD) |
| Limbic_OFC | +0.0005 (+ADHD) | +0.0001 (+ADHD) | +0.0049 (+ADHD) |
| Limbic_TempPole | -0.0004 (-ADHD) | +0.0001 (+ADHD) | +0.0145 (+ADHD) |
| SalVentAttnA | -0.0006 (-ADHD) | +0.0002 (+ADHD) | +0.0006 (+ADHD) |
| SalVentAttnB | -0.0006 (-ADHD) | +0.0002 (+ADHD) | +0.0011 (+ADHD) |
| SomMotA | -0.0013 (-ADHD) | -0.0002 (-ADHD) | +0.0004 (+ADHD) |
| SomMotB | -0.0009 (-ADHD) | +0.0000 (+ADHD) | +0.0001 (+ADHD) |
| TempPar | -0.0007 (-ADHD) | +0.0001 (+ADHD) | +0.0018 (+ADHD) |
| VisCent | -0.0008 (-ADHD) | -0.0001 (-ADHD) | +0.0014 (+ADHD) |
| VisPeri | -0.0003 (-ADHD) | +0.0001 (+ADHD) | +0.0013 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
