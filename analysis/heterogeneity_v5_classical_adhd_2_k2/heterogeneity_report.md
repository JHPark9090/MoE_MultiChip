# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_2 | **K**: 2 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.4885

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 115 | Internal | 0.546±0.062 | 0.454±0.062 |
| 1 | 173 | External | 0.441±0.055 | 0.559±0.055 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 227.213 | 0.0000 | Yes |
| External | 227.213 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.4296

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 127 | Internal | 5.910 | 1.784 |
| 1 | 161 | Internal | 3.797 | 1.748 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.4223

### Cluster 0 — Top 10 ROIs by Importance (N=228)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 20 | LO2 | VisCent | 0.7212 | -0.0005 | -ADHD |
| 2 | 171 | TGv | Limbic_TempPole | 0.7149 | +0.0005 | +ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.7076 | -0.0006 | -ADHD |
| 4 | 160 | 31pd | DefaultC | 0.6864 | -0.0002 | -ADHD |
| 5 | 168 | FOP5 | SalVentAttnB | 0.6686 | +0.0003 | +ADHD |
| 6 | 73 | 44 | TempPar | 0.6678 | -0.0003 | -ADHD |
| 7 | 38 | 5L | DorsAttnB | 0.6669 | -0.0005 | -ADHD |
| 8 | 174 | A4 | SomMotB | 0.6388 | -0.0005 | -ADHD |
| 9 | 169 | p10p | ContC | 0.6364 | +0.0002 | +ADHD |
| 10 | 135 | TE2p | Limbic_TempPole | 0.6230 | -0.0006 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0006 |
| 171 | TGv | Limbic_TempPole | +0.0005 |
| 91 | 13l | Limbic_OFC | +0.0004 |
| 165 | pOFC | Limbic_OFC | +0.0004 |
| 110 | AVI | SalVentAttnB | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 138 | TPOJ1 | DefaultA | -0.0007 |
| 136 | PHT | ContB | -0.0006 |
| 135 | TE2p | Limbic_TempPole | -0.0006 |
| 134 | TF | Limbic_TempPole | -0.0006 |
| 35 | 5m | SomMotA | -0.0006 |

### Cluster 1 — Top 10 ROIs by Importance (N=60)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 38 | 5L | DorsAttnB | 1.2586 | -0.0022 | -ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.1804 | +0.0033 | +ADHD |
| 3 | 171 | TGv | Limbic_TempPole | 1.1471 | +0.0157 | +ADHD |
| 4 | 161 | 31a | ContA | 1.1232 | -0.0014 | -ADHD |
| 5 | 54 | 6mp | SomMotA | 1.1182 | -0.0015 | -ADHD |
| 6 | 168 | FOP5 | SalVentAttnB | 1.0452 | -0.0005 | -ADHD |
| 7 | 60 | a24 | DefaultC | 1.0362 | +0.0012 | +ADHD |
| 8 | 160 | 31pd | DefaultC | 1.0342 | -0.0014 | -ADHD |
| 9 | 134 | TF | Limbic_TempPole | 1.0247 | +0.0083 | +ADHD |
| 10 | 165 | pOFC | Limbic_OFC | 1.0238 | +0.0036 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.0157 |
| 121 | PeEc | Limbic_TempPole | +0.0089 |
| 134 | TF | Limbic_TempPole | +0.0083 |
| 130 | TGd | Limbic_TempPole | +0.0073 |
| 117 | EC | Limbic_TempPole | +0.0054 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 38 | 5L | DorsAttnB | -0.0022 |
| 173 | LBelt | SomMotB | -0.0018 |
| 123 | PBelt | SomMotB | -0.0018 |
| 172 | MBelt | SomMotB | -0.0017 |
| 102 | 52 | SomMotB | -0.0017 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | 0.5017 | 0.8595 |
| ContB | 0.4441 | 0.6924 |
| ContC | 0.4821 | 0.7323 |
| DefaultA | 0.4826 | 0.6941 |
| DefaultB | 0.4907 | 0.7150 |
| DefaultC | 0.4798 | 0.7974 |
| DorsAttnA | 0.4502 | 0.6556 |
| DorsAttnB | 0.4534 | 0.7724 |
| Limbic_OFC | 0.4338 | 0.8760 |
| Limbic_TempPole | 0.5296 | 0.8665 |
| SalVentAttnA | 0.4458 | 0.7523 |
| SalVentAttnB | 0.4662 | 0.7645 |
| SomMotA | 0.4367 | 0.8192 |
| SomMotB | 0.4879 | 0.7647 |
| TempPar | 0.4652 | 0.7174 |
| VisCent | 0.4899 | 0.6175 |
| VisPeri | 0.4755 | 0.6345 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 |
|---------|:--------:|:--------:|
| ContA | -0.0002 (-ADHD) | -0.0007 (-ADHD) |
| ContB | -0.0003 (-ADHD) | -0.0007 (-ADHD) |
| ContC | -0.0002 (-ADHD) | -0.0003 (-ADHD) |
| DefaultA | -0.0004 (-ADHD) | +0.0000 (+ADHD) |
| DefaultB | -0.0002 (-ADHD) | +0.0003 (+ADHD) |
| DefaultC | -0.0001 (-ADHD) | -0.0005 (-ADHD) |
| DorsAttnA | -0.0003 (-ADHD) | +0.0000 (+ADHD) |
| DorsAttnB | -0.0002 (-ADHD) | -0.0012 (-ADHD) |
| Limbic_OFC | +0.0002 (+ADHD) | +0.0025 (+ADHD) |
| Limbic_TempPole | -0.0001 (-ADHD) | +0.0066 (+ADHD) |
| SalVentAttnA | +0.0001 (+ADHD) | -0.0005 (-ADHD) |
| SalVentAttnB | +0.0001 (+ADHD) | -0.0002 (-ADHD) |
| SomMotA | -0.0003 (-ADHD) | -0.0011 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | -0.0008 (-ADHD) |
| TempPar | -0.0001 (-ADHD) | +0.0002 (+ADHD) |
| VisCent | -0.0003 (-ADHD) | +0.0002 (+ADHD) |
| VisPeri | -0.0001 (-ADHD) | +0.0005 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
