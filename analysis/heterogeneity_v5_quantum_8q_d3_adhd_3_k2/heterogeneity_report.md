# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_3 | **K**: 2 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.8502

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 281 | SensoriMotor | 0.249±0.001 | 0.251±0.002 | 0.248±0.002 | 0.252±0.001 |
| 1 | 7 | Salience | 0.241±0.002 | 0.239±0.003 | 0.267±0.004 | 0.254±0.002 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 193.360 | 0.0000 | Yes |
| Executive | 303.861 | 0.0000 | Yes |
| Salience | 378.272 | 0.0000 | Yes |
| SensoriMotor | 28.554 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.2764

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 154 | Salience | 1.954 | 1.693 | 2.297 | 1.245 |
| 1 | 134 | Executive | 1.282 | 1.416 | 1.253 | 1.233 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.4259

### Cluster 0 — Top 10 ROIs by Importance (N=228)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 134 | TF | Limbic_TempPole | 0.5951 | -0.0005 | -ADHD |
| 2 | 32 | v23ab | DefaultC | 0.5902 | -0.0000 | -ADHD |
| 3 | 149 | PGi | DefaultC | 0.5779 | -0.0001 | -ADHD |
| 4 | 70 | 9p | TempPar | 0.5749 | +0.0001 | +ADHD |
| 5 | 175 | STSva | TempPar | 0.5749 | -0.0004 | -ADHD |
| 6 | 80 | IFSp | ContB | 0.5719 | -0.0003 | -ADHD |
| 7 | 128 | STSdp | DefaultA | 0.5714 | -0.0003 | -ADHD |
| 8 | 159 | VMV2 | VisPeri | 0.5610 | +0.0001 | +ADHD |
| 9 | 77 | 6r | ContB | 0.5599 | +0.0000 | +ADHD |
| 10 | 126 | PHA3 | DefaultB | 0.5545 | -0.0005 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0006 |
| 91 | 13l | Limbic_OFC | +0.0005 |
| 165 | pOFC | Limbic_OFC | +0.0004 |
| 110 | AVI | SalVentAttnB | +0.0004 |
| 112 | FOP1 | SalVentAttnA | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0007 |
| 136 | PHT | ContB | -0.0007 |
| 35 | 5m | SomMotA | -0.0006 |
| 138 | TPOJ1 | DefaultA | -0.0006 |
| 21 | PIT | VisCent | -0.0005 |

### Cluster 1 — Top 10 ROIs by Importance (N=60)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.1146 | +0.0031 | +ADHD |
| 2 | 165 | pOFC | Limbic_OFC | 1.0105 | +0.0036 | +ADHD |
| 3 | 39 | 24dd | SomMotA | 1.0075 | -0.0009 | -ADHD |
| 4 | 70 | 9p | TempPar | 0.9995 | -0.0011 | -ADHD |
| 5 | 163 | 25 | Limbic_OFC | 0.9834 | +0.0023 | +ADHD |
| 6 | 177 | PI | SalVentAttnA | 0.9543 | +0.0030 | +ADHD |
| 7 | 164 | s32 | Limbic_OFC | 0.9532 | +0.0015 | +ADHD |
| 8 | 54 | 6mp | SomMotA | 0.9351 | -0.0013 | -ADHD |
| 9 | 179 | p24 | DefaultC | 0.9234 | +0.0001 | +ADHD |
| 10 | 40 | 24dv | SalVentAttnA | 0.9141 | -0.0009 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.0114 |
| 121 | PeEc | Limbic_TempPole | +0.0091 |
| 130 | TGd | Limbic_TempPole | +0.0074 |
| 134 | TF | Limbic_TempPole | +0.0070 |
| 117 | EC | Limbic_TempPole | +0.0051 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0018 |
| 173 | LBelt | SomMotB | -0.0017 |
| 123 | PBelt | SomMotB | -0.0017 |
| 172 | MBelt | SomMotB | -0.0016 |
| 38 | 5L | DorsAttnB | -0.0016 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | 0.4456 | 0.7599 |
| ContB | 0.4916 | 0.7625 |
| ContC | 0.4827 | 0.7357 |
| DefaultA | 0.5043 | 0.7252 |
| DefaultB | 0.5025 | 0.7330 |
| DefaultC | 0.5006 | 0.8331 |
| DorsAttnA | 0.4742 | 0.6939 |
| DorsAttnB | 0.4610 | 0.7805 |
| Limbic_OFC | 0.4360 | 0.8883 |
| Limbic_TempPole | 0.4618 | 0.7573 |
| SalVentAttnA | 0.4608 | 0.7811 |
| SalVentAttnB | 0.4827 | 0.7943 |
| SomMotA | 0.4517 | 0.8443 |
| SomMotB | 0.4870 | 0.7690 |
| TempPar | 0.4931 | 0.7606 |
| VisCent | 0.4948 | 0.6241 |
| VisPeri | 0.4846 | 0.6458 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | -0.0002 (-ADHD) | -0.0006 (-ADHD) |
| ContB | -0.0003 (-ADHD) | -0.0008 (-ADHD) |
| ContC | -0.0002 (-ADHD) | -0.0002 (-ADHD) |
| DefaultA | -0.0004 (-ADHD) | +0.0001 (+ADHD) |
| DefaultB | -0.0002 (-ADHD) | +0.0003 (+ADHD) |
| DefaultC | -0.0001 (-ADHD) | -0.0005 (-ADHD) |
| DorsAttnA | -0.0003 (-ADHD) | +0.0000 (+ADHD) |
| DorsAttnB | -0.0002 (-ADHD) | -0.0011 (-ADHD) |
| Limbic_OFC | +0.0002 (+ADHD) | +0.0025 (+ADHD) |
| Limbic_TempPole | -0.0001 (-ADHD) | +0.0058 (+ADHD) |
| SalVentAttnA | +0.0001 (+ADHD) | -0.0005 (-ADHD) |
| SalVentAttnB | +0.0001 (+ADHD) | -0.0002 (-ADHD) |
| SomMotA | -0.0003 (-ADHD) | -0.0011 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | -0.0008 (-ADHD) |
| TempPar | -0.0001 (-ADHD) | +0.0002 (+ADHD) |
| VisCent | -0.0003 (-ADHD) | +0.0003 (+ADHD) |
| VisPeri | -0.0001 (-ADHD) | +0.0006 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
