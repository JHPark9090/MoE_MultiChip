# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_2 | **K**: 3 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.5401

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 214 | External | 0.496±0.035 | 0.504±0.035 |
| 1 | 60 | External | 0.388±0.064 | 0.612±0.064 |
| 2 | 14 | Internal | 0.685±0.077 | 0.315±0.077 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 279.054 | 0.0000 | Yes |
| External | 279.054 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3504

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 94 | Internal | 3.908 | 1.788 |
| 1 | 89 | Internal | 6.463 | 1.849 |
| 2 | 105 | Internal | 3.994 | 1.670 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2384

### Cluster 0 — Top 10 ROIs by Importance (N=94)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 38 | 5L | DorsAttnB | 0.9371 | -0.0021 | -ADHD |
| 2 | 171 | TGv | Limbic_TempPole | 0.8996 | +0.0007 | +ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.8687 | -0.0018 | -ADHD |
| 4 | 168 | FOP5 | SalVentAttnB | 0.8544 | -0.0009 | -ADHD |
| 5 | 73 | 44 | TempPar | 0.8465 | -0.0008 | -ADHD |
| 6 | 169 | p10p | ContC | 0.8366 | -0.0006 | -ADHD |
| 7 | 161 | 31a | ContA | 0.8341 | -0.0021 | -ADHD |
| 8 | 160 | 31pd | DefaultC | 0.8301 | -0.0023 | -ADHD |
| 9 | 54 | 6mp | SomMotA | 0.8133 | -0.0011 | -ADHD |
| 10 | 20 | LO2 | VisCent | 0.7877 | -0.0021 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0016 |
| 89 | 10pp | Limbic_OFC | +0.0007 |
| 171 | TGv | Limbic_TempPole | +0.0007 |
| 91 | 13l | Limbic_OFC | +0.0006 |
| 165 | pOFC | Limbic_OFC | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 160 | 31pd | DefaultC | -0.0023 |
| 161 | 31a | ContA | -0.0021 |
| 20 | LO2 | VisCent | -0.0021 |
| 38 | 5L | DorsAttnB | -0.0021 |
| 134 | TF | Limbic_TempPole | -0.0018 |

### Cluster 1 — Top 10 ROIs by Importance (N=170)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 20 | LO2 | VisCent | 0.7107 | -0.0001 | -ADHD |
| 2 | 171 | TGv | Limbic_TempPole | 0.6655 | +0.0004 | +ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.6614 | +0.0001 | +ADHD |
| 4 | 160 | 31pd | DefaultC | 0.6526 | +0.0002 | +ADHD |
| 5 | 168 | FOP5 | SalVentAttnB | 0.6263 | +0.0005 | +ADHD |
| 6 | 73 | 44 | TempPar | 0.6208 | -0.0000 | -ADHD |
| 7 | 38 | 5L | DorsAttnB | 0.6083 | -0.0004 | -ADHD |
| 8 | 174 | A4 | SomMotB | 0.6029 | -0.0004 | -ADHD |
| 9 | 169 | p10p | ContC | 0.5885 | +0.0005 | +ADHD |
| 10 | 32 | v23ab | DefaultC | 0.5848 | +0.0002 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0010 |
| 119 | H | Limbic_TempPole | +0.0007 |
| 167 | Ig | SomMotB | +0.0006 |
| 120 | ProS | VisPeri | +0.0005 |
| 168 | FOP5 | SalVentAttnB | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 144 | IP1 | ContB | -0.0007 |
| 135 | TE2p | Limbic_TempPole | -0.0006 |
| 18 | V3B | VisCent | -0.0005 |
| 136 | PHT | ContB | -0.0005 |
| 138 | TPOJ1 | DefaultA | -0.0005 |

### Cluster 2 — Top 10 ROIs by Importance (N=24)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 38 | 5L | DorsAttnB | 1.5032 | +0.0008 | +ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.5012 | +0.0036 | +ADHD |
| 3 | 171 | TGv | Limbic_TempPole | 1.4223 | +0.0386 | +ADHD |
| 4 | 161 | 31a | ContA | 1.3699 | +0.0018 | +ADHD |
| 5 | 60 | a24 | DefaultC | 1.3440 | +0.0052 | +ADHD |
| 6 | 54 | 6mp | SomMotA | 1.3414 | +0.0002 | +ADHD |
| 7 | 165 | pOFC | Limbic_OFC | 1.3157 | +0.0081 | +ADHD |
| 8 | 177 | PI | SalVentAttnA | 1.2800 | +0.0094 | +ADHD |
| 9 | 89 | 10pp | Limbic_OFC | 1.2507 | +0.0038 | +ADHD |
| 10 | 160 | 31pd | DefaultC | 1.2320 | +0.0013 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.0386 |
| 121 | PeEc | Limbic_TempPole | +0.0231 |
| 134 | TF | Limbic_TempPole | +0.0219 |
| 130 | TGd | Limbic_TempPole | +0.0169 |
| 117 | EC | Limbic_TempPole | +0.0135 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0018 |
| 123 | PBelt | SomMotB | -0.0012 |
| 172 | MBelt | SomMotB | -0.0010 |
| 66 | 8Av | ContC | -0.0008 |
| 99 | OP4 | SomMotB | -0.0008 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | 0.6514 | 0.4675 | 1.0513 |
| ContB | 0.5660 | 0.4147 | 0.7957 |
| ContC | 0.6055 | 0.4519 | 0.8376 |
| DefaultA | 0.6024 | 0.4491 | 0.7800 |
| DefaultB | 0.6041 | 0.4632 | 0.8016 |
| DefaultC | 0.6169 | 0.4479 | 0.9621 |
| DorsAttnA | 0.5390 | 0.4271 | 0.7801 |
| DorsAttnB | 0.5996 | 0.4201 | 0.9141 |
| Limbic_OFC | 0.5961 | 0.3995 | 1.1467 |
| Limbic_TempPole | 0.6776 | 0.4935 | 1.0483 |
| SalVentAttnA | 0.5878 | 0.4133 | 0.8867 |
| SalVentAttnB | 0.6097 | 0.4334 | 0.8826 |
| SomMotA | 0.6168 | 0.3947 | 0.9851 |
| SomMotB | 0.6266 | 0.4541 | 0.8758 |
| TempPar | 0.5870 | 0.4365 | 0.8218 |
| VisCent | 0.5436 | 0.4768 | 0.6917 |
| VisPeri | 0.5434 | 0.4596 | 0.7199 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | -0.0012 (-ADHD) | -0.0001 (-ADHD) | +0.0010 (+ADHD) |
| ContB | -0.0008 (-ADHD) | -0.0001 (-ADHD) | +0.0001 (+ADHD) |
| ContC | -0.0010 (-ADHD) | +0.0001 (+ADHD) | +0.0011 (+ADHD) |
| DefaultA | -0.0010 (-ADHD) | -0.0002 (-ADHD) | +0.0019 (+ADHD) |
| DefaultB | -0.0007 (-ADHD) | +0.0000 (+ADHD) | +0.0015 (+ADHD) |
| DefaultC | -0.0009 (-ADHD) | +0.0000 (+ADHD) | +0.0009 (+ADHD) |
| DorsAttnA | -0.0006 (-ADHD) | -0.0003 (-ADHD) | +0.0013 (+ADHD) |
| DorsAttnB | -0.0012 (-ADHD) | -0.0000 (-ADHD) | +0.0001 (+ADHD) |
| Limbic_OFC | +0.0006 (+ADHD) | +0.0001 (+ADHD) | +0.0048 (+ADHD) |
| Limbic_TempPole | -0.0004 (-ADHD) | +0.0000 (+ADHD) | +0.0166 (+ADHD) |
| SalVentAttnA | -0.0006 (-ADHD) | +0.0002 (+ADHD) | +0.0006 (+ADHD) |
| SalVentAttnB | -0.0006 (-ADHD) | +0.0002 (+ADHD) | +0.0011 (+ADHD) |
| SomMotA | -0.0012 (-ADHD) | -0.0002 (-ADHD) | +0.0004 (+ADHD) |
| SomMotB | -0.0009 (-ADHD) | +0.0001 (+ADHD) | +0.0001 (+ADHD) |
| TempPar | -0.0007 (-ADHD) | +0.0001 (+ADHD) | +0.0017 (+ADHD) |
| VisCent | -0.0008 (-ADHD) | -0.0001 (-ADHD) | +0.0013 (+ADHD) |
| VisPeri | -0.0003 (-ADHD) | +0.0001 (+ADHD) | +0.0012 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
