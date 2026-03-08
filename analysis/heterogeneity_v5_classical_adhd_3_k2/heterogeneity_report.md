# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_3 | **K**: 2 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.8086

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 280 | Executive | 0.250±0.001 | 0.252±0.000 | 0.248±0.001 | 0.251±0.001 |
| 1 | 8 | DMN | 0.254±0.002 | 0.252±0.001 | 0.250±0.002 | 0.244±0.003 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 179.284 | 0.0000 | Yes |
| Executive | 1.815 | 0.1790 | No |
| Salience | 51.622 | 0.0000 | Yes |
| SensoriMotor | 318.265 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.6646

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 260 | DMN | 3.858 | 3.386 | 3.594 | 2.420 |
| 1 | 28 | DMN | 3.512 | 3.315 | 3.180 | 2.372 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.4290

### Cluster 0 — Top 10 ROIs by Importance (N=230)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5918 | -0.0000 | -ADHD |
| 2 | 15 | V7 | VisCent | 0.5658 | -0.0003 | -ADHD |
| 3 | 157 | V3CD | VisCent | 0.5540 | -0.0005 | -ADHD |
| 4 | 20 | LO2 | VisCent | 0.5529 | -0.0004 | -ADHD |
| 5 | 175 | STSva | TempPar | 0.5485 | -0.0004 | -ADHD |
| 6 | 176 | TE1m | TempPar | 0.5462 | +0.0000 | +ADHD |
| 7 | 65 | 47m | TempPar | 0.5449 | -0.0001 | -ADHD |
| 8 | 126 | PHA3 | DefaultB | 0.5405 | -0.0005 | -ADHD |
| 9 | 124 | A5 | DefaultA | 0.5330 | -0.0002 | -ADHD |
| 10 | 3 | V2 | VisPeri | 0.5321 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0006 |
| 91 | 13l | Limbic_OFC | +0.0006 |
| 177 | PI | SalVentAttnA | +0.0005 |
| 89 | 10pp | Limbic_OFC | +0.0004 |
| 165 | pOFC | Limbic_OFC | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0007 |
| 136 | PHT | ContB | -0.0007 |
| 35 | 5m | SomMotA | -0.0006 |
| 138 | TPOJ1 | DefaultA | -0.0006 |
| 52 | 3a | SomMotA | -0.0006 |

### Cluster 1 — Top 10 ROIs by Importance (N=58)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 35 | 5m | SomMotA | 0.9702 | -0.0016 | -ADHD |
| 2 | 163 | 25 | Limbic_OFC | 0.9412 | +0.0023 | +ADHD |
| 3 | 92 | OFC | Limbic_OFC | 0.9399 | +0.0013 | +ADHD |
| 4 | 52 | 3a | SomMotA | 0.9320 | -0.0010 | -ADHD |
| 5 | 91 | 13l | Limbic_OFC | 0.9145 | +0.0044 | +ADHD |
| 6 | 165 | pOFC | Limbic_OFC | 0.9097 | +0.0033 | +ADHD |
| 7 | 37 | 23c | SalVentAttnA | 0.9080 | -0.0008 | -ADHD |
| 8 | 31 | 23d | DefaultC | 0.9005 | -0.0006 | -ADHD |
| 9 | 60 | a24 | DefaultC | 0.8936 | +0.0013 | +ADHD |
| 10 | 166 | PoI1 | SalVentAttnA | 0.8863 | -0.0012 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.0103 |
| 171 | TGv | Limbic_TempPole | +0.0098 |
| 130 | TGd | Limbic_TempPole | +0.0081 |
| 134 | TF | Limbic_TempPole | +0.0064 |
| 117 | EC | Limbic_TempPole | +0.0047 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0017 |
| 172 | MBelt | SomMotB | -0.0016 |
| 173 | LBelt | SomMotB | -0.0016 |
| 35 | 5m | SomMotA | -0.0016 |
| 143 | IP2 | ContB | -0.0015 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | 0.4339 | 0.7448 |
| ContB | 0.4456 | 0.6970 |
| ContC | 0.4171 | 0.6338 |
| DefaultA | 0.5068 | 0.7334 |
| DefaultB | 0.4909 | 0.7186 |
| DefaultC | 0.4852 | 0.8148 |
| DorsAttnA | 0.4528 | 0.6677 |
| DorsAttnB | 0.4319 | 0.7329 |
| Limbic_OFC | 0.4352 | 0.8706 |
| Limbic_TempPole | 0.4510 | 0.7391 |
| SalVentAttnA | 0.4549 | 0.7715 |
| SalVentAttnB | 0.4684 | 0.7759 |
| SomMotA | 0.4665 | 0.8716 |
| SomMotB | 0.4733 | 0.7520 |
| TempPar | 0.4899 | 0.7579 |
| VisCent | 0.5117 | 0.6470 |
| VisPeri | 0.4847 | 0.6525 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | -0.0002 (-ADHD) | -0.0006 (-ADHD) |
| ContB | -0.0003 (-ADHD) | -0.0007 (-ADHD) |
| ContC | -0.0002 (-ADHD) | -0.0002 (-ADHD) |
| DefaultA | -0.0004 (-ADHD) | +0.0002 (+ADHD) |
| DefaultB | -0.0002 (-ADHD) | +0.0003 (+ADHD) |
| DefaultC | -0.0001 (-ADHD) | -0.0005 (-ADHD) |
| DorsAttnA | -0.0003 (-ADHD) | +0.0000 (+ADHD) |
| DorsAttnB | -0.0002 (-ADHD) | -0.0011 (-ADHD) |
| Limbic_OFC | +0.0003 (+ADHD) | +0.0022 (+ADHD) |
| Limbic_TempPole | -0.0001 (-ADHD) | +0.0058 (+ADHD) |
| SalVentAttnA | +0.0001 (+ADHD) | -0.0005 (-ADHD) |
| SalVentAttnB | +0.0001 (+ADHD) | -0.0002 (-ADHD) |
| SomMotA | -0.0004 (-ADHD) | -0.0012 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | -0.0007 (-ADHD) |
| TempPar | -0.0000 (-ADHD) | +0.0002 (+ADHD) |
| VisCent | -0.0003 (-ADHD) | +0.0003 (+ADHD) |
| VisPeri | -0.0001 (-ADHD) | +0.0006 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
