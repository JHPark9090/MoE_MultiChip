# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_2 | **K**: 2 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.8656

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 280 | External | 0.496±0.014 | 0.504±0.014 |
| 1 | 8 | Internal | 0.616±0.033 | 0.384±0.033 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 491.468 | 0.0000 | Yes |
| External | 491.468 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3129

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 126 | Internal | 2.662 | 1.439 |
| 1 | 162 | External | 1.385 | 1.652 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.4251

### Cluster 0 — Top 10 ROIs by Importance (N=228)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 128 | STSdp | DefaultA | 0.4949 | -0.0003 | -ADHD |
| 2 | 100 | OP1 | SomMotB | 0.4720 | +0.0004 | +ADHD |
| 3 | 131 | TE1a | TempPar | 0.4716 | +0.0002 | +ADHD |
| 4 | 174 | A4 | SomMotB | 0.4670 | -0.0004 | -ADHD |
| 5 | 157 | V3CD | VisCent | 0.4637 | -0.0004 | -ADHD |
| 6 | 76 | a47r | ContC | 0.4606 | -0.0002 | -ADHD |
| 7 | 18 | V3B | VisCent | 0.4594 | -0.0003 | -ADHD |
| 8 | 1 | MST | DorsAttnA | 0.4586 | -0.0002 | -ADHD |
| 9 | 16 | IPS1 | DorsAttnA | 0.4560 | -0.0003 | -ADHD |
| 10 | 115 | PFt | DorsAttnB | 0.4516 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0005 |
| 91 | 13l | Limbic_OFC | +0.0004 |
| 100 | OP1 | SomMotB | +0.0004 |
| 110 | AVI | SalVentAttnB | +0.0003 |
| 165 | pOFC | Limbic_OFC | +0.0003 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0006 |
| 136 | PHT | ContB | -0.0005 |
| 138 | TPOJ1 | DefaultA | -0.0005 |
| 137 | PH | DorsAttnA | -0.0005 |
| 17 | FFC | DorsAttnA | -0.0005 |

### Cluster 1 — Top 10 ROIs by Importance (N=60)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 0.9380 | +0.0026 | +ADHD |
| 2 | 54 | 6mp | SomMotA | 0.8218 | -0.0011 | -ADHD |
| 3 | 177 | PI | SalVentAttnA | 0.7980 | +0.0025 | +ADHD |
| 4 | 100 | OP1 | SomMotB | 0.7933 | -0.0006 | -ADHD |
| 5 | 41 | 7AL | DorsAttnB | 0.7836 | -0.0012 | -ADHD |
| 6 | 164 | s32 | Limbic_OFC | 0.7709 | +0.0012 | +ADHD |
| 7 | 96 | i6-8 | ContC | 0.7644 | -0.0013 | -ADHD |
| 8 | 161 | 31a | ContA | 0.7611 | -0.0009 | -ADHD |
| 9 | 106 | TA2 | SomMotB | 0.7560 | +0.0013 | +ADHD |
| 10 | 10 | PEF | ContB | 0.7539 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.0084 |
| 171 | TGv | Limbic_TempPole | +0.0076 |
| 130 | TGd | Limbic_TempPole | +0.0065 |
| 134 | TF | Limbic_TempPole | +0.0042 |
| 117 | EC | Limbic_TempPole | +0.0039 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0016 |
| 62 | 8BM | ContC | -0.0013 |
| 173 | LBelt | SomMotB | -0.0013 |
| 123 | PBelt | SomMotB | -0.0013 |
| 115 | PFt | DorsAttnB | -0.0013 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | 0.3853 | 0.6572 |
| ContB | 0.4058 | 0.6314 |
| ContC | 0.4044 | 0.6166 |
| DefaultA | 0.4002 | 0.5752 |
| DefaultB | 0.3808 | 0.5569 |
| DefaultC | 0.3488 | 0.5805 |
| DorsAttnA | 0.4233 | 0.6193 |
| DorsAttnB | 0.4020 | 0.6790 |
| Limbic_OFC | 0.3437 | 0.6986 |
| Limbic_TempPole | 0.3751 | 0.6184 |
| SalVentAttnA | 0.3654 | 0.6174 |
| SalVentAttnB | 0.3805 | 0.6253 |
| SomMotA | 0.3696 | 0.6901 |
| SomMotB | 0.4070 | 0.6448 |
| TempPar | 0.3714 | 0.5727 |
| VisCent | 0.4236 | 0.5345 |
| VisPeri | 0.3719 | 0.4963 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | -0.0002 (-ADHD) | -0.0006 (-ADHD) |
| ContB | -0.0002 (-ADHD) | -0.0006 (-ADHD) |
| ContC | -0.0002 (-ADHD) | -0.0003 (-ADHD) |
| DefaultA | -0.0003 (-ADHD) | +0.0001 (+ADHD) |
| DefaultB | -0.0002 (-ADHD) | +0.0002 (+ADHD) |
| DefaultC | -0.0001 (-ADHD) | -0.0004 (-ADHD) |
| DorsAttnA | -0.0003 (-ADHD) | +0.0000 (+ADHD) |
| DorsAttnB | -0.0002 (-ADHD) | -0.0010 (-ADHD) |
| Limbic_OFC | +0.0001 (+ADHD) | +0.0020 (+ADHD) |
| Limbic_TempPole | -0.0001 (-ADHD) | +0.0046 (+ADHD) |
| SalVentAttnA | +0.0001 (+ADHD) | -0.0004 (-ADHD) |
| SalVentAttnB | +0.0001 (+ADHD) | -0.0001 (-ADHD) |
| SomMotA | -0.0003 (-ADHD) | -0.0009 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | -0.0006 (-ADHD) |
| TempPar | -0.0000 (-ADHD) | +0.0002 (+ADHD) |
| VisCent | -0.0003 (-ADHD) | +0.0002 (+ADHD) |
| VisPeri | -0.0001 (-ADHD) | +0.0004 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
