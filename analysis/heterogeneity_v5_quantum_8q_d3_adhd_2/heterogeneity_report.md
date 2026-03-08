# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_2 | **K**: 3 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.5101

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 146 | External | 0.486±0.011 | 0.514±0.011 |
| 1 | 7 | Internal | 0.624±0.027 | 0.376±0.027 |
| 2 | 135 | Internal | 0.507±0.010 | 0.493±0.010 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 566.106 | 0.0000 | Yes |
| External | 566.105 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.2090

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 79 | Internal | 3.024 | 1.488 |
| 1 | 91 | External | 1.252 | 1.759 |
| 2 | 118 | Internal | 1.753 | 1.453 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2307

### Cluster 0 — Top 10 ROIs by Importance (N=27)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.1207 | +0.0026 | +ADHD |
| 2 | 164 | s32 | Limbic_OFC | 1.0058 | +0.0017 | +ADHD |
| 3 | 177 | PI | SalVentAttnA | 0.9948 | +0.0057 | +ADHD |
| 4 | 54 | 6mp | SomMotA | 0.9850 | -0.0009 | -ADHD |
| 5 | 163 | 25 | Limbic_OFC | 0.9679 | +0.0026 | +ADHD |
| 6 | 41 | 7AL | DorsAttnB | 0.9469 | -0.0005 | -ADHD |
| 7 | 100 | OP1 | SomMotB | 0.9289 | -0.0011 | -ADHD |
| 8 | 161 | 31a | ContA | 0.9245 | +0.0004 | +ADHD |
| 9 | 106 | TA2 | SomMotB | 0.9108 | +0.0030 | +ADHD |
| 10 | 7 | 4 | SomMotA | 0.8840 | -0.0006 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.0186 |
| 171 | TGv | Limbic_TempPole | +0.0164 |
| 130 | TGd | Limbic_TempPole | +0.0127 |
| 134 | TF | Limbic_TempPole | +0.0092 |
| 109 | Pir | Limbic_TempPole | +0.0079 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0022 |
| 123 | PBelt | SomMotB | -0.0017 |
| 79 | IFJp | ContB | -0.0016 |
| 172 | MBelt | SomMotB | -0.0015 |
| 173 | LBelt | SomMotB | -0.0015 |

### Cluster 1 — Top 10 ROIs by Importance (N=96)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 100 | OP1 | SomMotB | 0.6055 | +0.0002 | +ADHD |
| 2 | 128 | STSdp | DefaultA | 0.6055 | -0.0006 | -ADHD |
| 3 | 92 | OFC | Limbic_OFC | 0.5984 | +0.0010 | +ADHD |
| 4 | 41 | 7AL | DorsAttnB | 0.5831 | -0.0012 | -ADHD |
| 5 | 10 | PEF | ContB | 0.5812 | -0.0001 | -ADHD |
| 6 | 96 | i6-8 | ContC | 0.5808 | -0.0010 | -ADHD |
| 7 | 54 | 6mp | SomMotA | 0.5800 | -0.0007 | -ADHD |
| 8 | 174 | A4 | SomMotB | 0.5690 | -0.0004 | -ADHD |
| 9 | 1 | MST | DorsAttnA | 0.5585 | -0.0005 | -ADHD |
| 10 | 81 | IFSa | ContB | 0.5576 | -0.0006 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0010 |
| 154 | PHA2 | DefaultB | +0.0007 |
| 89 | 10pp | Limbic_OFC | +0.0005 |
| 165 | pOFC | Limbic_OFC | +0.0005 |
| 125 | PHA1 | DefaultB | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 41 | 7AL | DorsAttnB | -0.0012 |
| 62 | 8BM | ContC | -0.0011 |
| 35 | 5m | SomMotA | -0.0011 |
| 76 | a47r | ContC | -0.0011 |
| 38 | 5L | DorsAttnB | -0.0011 |

### Cluster 2 — Top 10 ROIs by Importance (N=165)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 157 | V3CD | VisCent | 0.4597 | -0.0003 | -ADHD |
| 2 | 128 | STSdp | DefaultA | 0.4567 | -0.0002 | -ADHD |
| 3 | 131 | TE1a | TempPar | 0.4497 | +0.0001 | +ADHD |
| 4 | 18 | V3B | VisCent | 0.4494 | -0.0005 | -ADHD |
| 5 | 76 | a47r | ContC | 0.4438 | +0.0002 | +ADHD |
| 6 | 174 | A4 | SomMotB | 0.4386 | -0.0003 | -ADHD |
| 7 | 100 | OP1 | SomMotB | 0.4364 | +0.0003 | +ADHD |
| 8 | 21 | PIT | VisCent | 0.4361 | -0.0004 | -ADHD |
| 9 | 16 | IPS1 | DorsAttnA | 0.4322 | -0.0004 | -ADHD |
| 10 | 1 | MST | DorsAttnA | 0.4258 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0006 |
| 120 | ProS | VisPeri | +0.0004 |
| 91 | 13l | Limbic_OFC | +0.0004 |
| 127 | STSda | TempPar | +0.0004 |
| 100 | OP1 | SomMotB | +0.0003 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 126 | PHA3 | DefaultB | -0.0006 |
| 144 | IP1 | ContB | -0.0005 |
| 135 | TE2p | Limbic_TempPole | -0.0005 |
| 28 | 7Pm | ContA | -0.0005 |
| 18 | V3B | VisCent | -0.0005 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | 0.8007 | 0.4857 | 0.3577 |
| ContB | 0.7119 | 0.5102 | 0.3769 |
| ContC | 0.6916 | 0.5038 | 0.3768 |
| DefaultA | 0.6319 | 0.4963 | 0.3701 |
| DefaultB | 0.6180 | 0.4667 | 0.3561 |
| DefaultC | 0.6869 | 0.4409 | 0.3242 |
| DorsAttnA | 0.7267 | 0.5006 | 0.4000 |
| DorsAttnB | 0.7904 | 0.5215 | 0.3697 |
| Limbic_OFC | 0.8649 | 0.4705 | 0.3137 |
| Limbic_TempPole | 0.7273 | 0.4764 | 0.3471 |
| SalVentAttnA | 0.7164 | 0.4732 | 0.3368 |
| SalVentAttnB | 0.7054 | 0.4912 | 0.3520 |
| SomMotA | 0.8178 | 0.5085 | 0.3320 |
| SomMotB | 0.7326 | 0.5162 | 0.3767 |
| TempPar | 0.6413 | 0.4635 | 0.3468 |
| VisCent | 0.5901 | 0.4654 | 0.4123 |
| VisPeri | 0.5571 | 0.4204 | 0.3586 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | -0.0001 (-ADHD) | -0.0004 (-ADHD) | -0.0002 (-ADHD) |
| ContB | -0.0008 (-ADHD) | -0.0004 (-ADHD) | -0.0002 (-ADHD) |
| ContC | +0.0003 (+ADHD) | -0.0007 (-ADHD) | +0.0000 (+ADHD) |
| DefaultA | +0.0008 (+ADHD) | -0.0004 (-ADHD) | -0.0003 (-ADHD) |
| DefaultB | +0.0003 (+ADHD) | +0.0002 (+ADHD) | -0.0003 (-ADHD) |
| DefaultC | -0.0002 (-ADHD) | -0.0004 (-ADHD) | +0.0000 (+ADHD) |
| DorsAttnA | +0.0005 (+ADHD) | -0.0003 (-ADHD) | -0.0003 (-ADHD) |
| DorsAttnB | -0.0008 (-ADHD) | -0.0006 (-ADHD) | -0.0001 (-ADHD) |
| Limbic_OFC | +0.0030 (+ADHD) | +0.0005 (+ADHD) | +0.0001 (+ADHD) |
| Limbic_TempPole | +0.0094 (+ADHD) | +0.0001 (+ADHD) | -0.0001 (-ADHD) |
| SalVentAttnA | -0.0004 (-ADHD) | -0.0001 (-ADHD) | +0.0001 (+ADHD) |
| SalVentAttnB | +0.0001 (+ADHD) | -0.0002 (-ADHD) | +0.0002 (+ADHD) |
| SomMotA | -0.0006 (-ADHD) | -0.0007 (-ADHD) | -0.0002 (-ADHD) |
| SomMotB | -0.0009 (-ADHD) | -0.0002 (-ADHD) | -0.0001 (-ADHD) |
| TempPar | +0.0007 (+ADHD) | -0.0003 (-ADHD) | +0.0001 (+ADHD) |
| VisCent | +0.0009 (+ADHD) | -0.0005 (-ADHD) | -0.0002 (-ADHD) |
| VisPeri | +0.0006 (+ADHD) | -0.0000 (-ADHD) | +0.0000 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
