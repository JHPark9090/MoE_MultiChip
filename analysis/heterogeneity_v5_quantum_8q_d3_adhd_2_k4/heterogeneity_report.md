# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_2 | **K**: 4 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.5329

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 174 | External | 0.498±0.006 | 0.502±0.006 |
| 1 | 7 | Internal | 0.624±0.027 | 0.376±0.027 |
| 2 | 72 | External | 0.479±0.012 | 0.521±0.012 |
| 3 | 35 | Internal | 0.519±0.011 | 0.481±0.011 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 585.407 | 0.0000 | Yes |
| External | 585.407 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.1940

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 56 | External | 1.478 | 2.081 |
| 1 | 95 | Internal | 1.885 | 1.288 |
| 2 | 68 | External | 1.230 | 1.536 |
| 3 | 69 | Internal | 3.103 | 1.531 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2168

### Cluster 0 — Top 10 ROIs by Importance (N=94)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 128 | STSdp | DefaultA | 0.6038 | -0.0002 | -ADHD |
| 2 | 100 | OP1 | SomMotB | 0.5950 | +0.0008 | +ADHD |
| 3 | 174 | A4 | SomMotB | 0.5733 | -0.0003 | -ADHD |
| 4 | 41 | 7AL | DorsAttnB | 0.5630 | -0.0008 | -ADHD |
| 5 | 1 | MST | DorsAttnA | 0.5591 | -0.0003 | -ADHD |
| 6 | 10 | PEF | ContB | 0.5530 | +0.0004 | +ADHD |
| 7 | 17 | FFC | DorsAttnA | 0.5495 | -0.0002 | -ADHD |
| 8 | 54 | 6mp | SomMotA | 0.5486 | -0.0002 | -ADHD |
| 9 | 49 | MIP | DorsAttnA | 0.5486 | -0.0002 | -ADHD |
| 10 | 135 | TE2p | Limbic_TempPole | 0.5482 | -0.0001 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 111 | AAIC | SalVentAttnB | +0.0013 |
| 92 | OFC | Limbic_OFC | +0.0012 |
| 177 | PI | SalVentAttnA | +0.0012 |
| 113 | FOP3 | SalVentAttnA | +0.0010 |
| 110 | AVI | SalVentAttnB | +0.0010 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 155 | V4t | VisCent | -0.0010 |
| 20 | LO2 | VisCent | -0.0010 |
| 162 | VVC | VisCent | -0.0010 |
| 160 | 31pd | DefaultC | -0.0009 |
| 142 | PGp | DorsAttnA | -0.0009 |

### Cluster 1 — Top 10 ROIs by Importance (N=159)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 157 | V3CD | VisCent | 0.4574 | -0.0002 | -ADHD |
| 2 | 128 | STSdp | DefaultA | 0.4540 | -0.0003 | -ADHD |
| 3 | 18 | V3B | VisCent | 0.4473 | -0.0004 | -ADHD |
| 4 | 76 | a47r | ContC | 0.4469 | +0.0002 | +ADHD |
| 5 | 131 | TE1a | TempPar | 0.4457 | +0.0001 | +ADHD |
| 6 | 100 | OP1 | SomMotB | 0.4335 | +0.0002 | +ADHD |
| 7 | 174 | A4 | SomMotB | 0.4319 | -0.0004 | -ADHD |
| 8 | 21 | PIT | VisCent | 0.4315 | -0.0004 | -ADHD |
| 9 | 16 | IPS1 | DorsAttnA | 0.4296 | -0.0004 | -ADHD |
| 10 | 113 | FOP3 | SalVentAttnA | 0.4233 | -0.0001 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 120 | ProS | VisPeri | +0.0005 |
| 91 | 13l | Limbic_OFC | +0.0003 |
| 133 | TE2a | Limbic_TempPole | +0.0003 |
| 86 | 9a | TempPar | +0.0003 |
| 119 | H | Limbic_TempPole | +0.0003 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0005 |
| 106 | TA2 | SomMotB | -0.0005 |
| 144 | IP1 | ContB | -0.0005 |
| 28 | 7Pm | ContA | -0.0005 |
| 17 | FFC | DorsAttnA | -0.0005 |

### Cluster 2 — Top 10 ROIs by Importance (N=29)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.0279 | +0.0008 | +ADHD |
| 2 | 54 | 6mp | SomMotA | 0.8791 | -0.0022 | -ADHD |
| 3 | 164 | s32 | Limbic_OFC | 0.8635 | +0.0006 | +ADHD |
| 4 | 96 | i6-8 | ContC | 0.8392 | -0.0026 | -ADHD |
| 5 | 100 | OP1 | SomMotB | 0.8379 | -0.0015 | -ADHD |
| 6 | 41 | 7AL | DorsAttnB | 0.8308 | -0.0017 | -ADHD |
| 7 | 10 | PEF | ContB | 0.8280 | -0.0017 | -ADHD |
| 8 | 177 | PI | SalVentAttnA | 0.8119 | -0.0009 | -ADHD |
| 9 | 7 | 4 | SomMotA | 0.8089 | -0.0018 | -ADHD |
| 10 | 161 | 31a | ContA | 0.7948 | -0.0011 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 91 | 13l | Limbic_OFC | +0.0024 |
| 130 | TGd | Limbic_TempPole | +0.0012 |
| 88 | a10p | ContC | +0.0011 |
| 165 | pOFC | Limbic_OFC | +0.0011 |
| 163 | 25 | Limbic_OFC | +0.0010 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 11 | 55b | SomMotB | -0.0026 |
| 96 | i6-8 | ContC | -0.0026 |
| 115 | PFt | DorsAttnB | -0.0025 |
| 66 | 8Av | ContC | -0.0025 |
| 43 | 6ma | SalVentAttnA | -0.0022 |

### Cluster 3 — Top 10 ROIs by Importance (N=6)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 177 | PI | SalVentAttnA | 1.5874 | +0.0240 | +ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.5692 | +0.0039 | +ADHD |
| 3 | 163 | 25 | Limbic_OFC | 1.4959 | +0.0067 | +ADHD |
| 4 | 164 | s32 | Limbic_OFC | 1.3761 | +0.0027 | +ADHD |
| 5 | 106 | TA2 | SomMotB | 1.3755 | +0.0183 | +ADHD |
| 6 | 165 | pOFC | Limbic_OFC | 1.2907 | +0.0121 | +ADHD |
| 7 | 54 | 6mp | SomMotA | 1.2875 | -0.0008 | -ADHD |
| 8 | 161 | 31a | ContA | 1.2196 | +0.0010 | +ADHD |
| 9 | 89 | 10pp | Limbic_OFC | 1.2161 | +0.0044 | +ADHD |
| 10 | 111 | AAIC | SalVentAttnB | 1.1978 | +0.0120 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.0856 |
| 171 | TGv | Limbic_TempPole | +0.0720 |
| 130 | TGd | Limbic_TempPole | +0.0538 |
| 134 | TF | Limbic_TempPole | +0.0451 |
| 117 | EC | Limbic_TempPole | +0.0349 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0039 |
| 101 | OP2-3 | SomMotB | -0.0039 |
| 114 | FOP2 | SomMotB | -0.0036 |
| 69 | 8BL | TempPar | -0.0033 |
| 62 | 8BM | ContC | -0.0029 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.4768 | 0.3540 | 0.6780 | 1.0836 |
| ContB | 0.4990 | 0.3744 | 0.6561 | 0.8229 |
| ContC | 0.4890 | 0.3761 | 0.6297 | 0.8629 |
| DefaultA | 0.4929 | 0.3670 | 0.5597 | 0.8078 |
| DefaultB | 0.4641 | 0.3533 | 0.5556 | 0.7228 |
| DefaultC | 0.4290 | 0.3228 | 0.6038 | 0.8673 |
| DorsAttnA | 0.4984 | 0.3967 | 0.6256 | 0.9340 |
| DorsAttnB | 0.5037 | 0.3674 | 0.7187 | 0.9642 |
| Limbic_OFC | 0.4416 | 0.3136 | 0.7299 | 1.2907 |
| Limbic_TempPole | 0.4684 | 0.3442 | 0.6105 | 1.0285 |
| SalVentAttnA | 0.4590 | 0.3347 | 0.6488 | 0.8618 |
| SalVentAttnB | 0.4754 | 0.3496 | 0.6532 | 0.8434 |
| SomMotA | 0.4893 | 0.3289 | 0.7317 | 1.0300 |
| SomMotB | 0.5086 | 0.3730 | 0.6600 | 0.8716 |
| TempPar | 0.4498 | 0.3453 | 0.5929 | 0.7756 |
| VisCent | 0.4688 | 0.4086 | 0.5299 | 0.7074 |
| VisPeri | 0.4232 | 0.3552 | 0.4925 | 0.6741 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | -0.0002 (-ADHD) | -0.0002 (-ADHD) | -0.0011 (-ADHD) | +0.0009 (+ADHD) |
| ContB | -0.0000 (-ADHD) | -0.0002 (-ADHD) | -0.0015 (-ADHD) | -0.0006 (-ADHD) |
| ContC | -0.0003 (-ADHD) | -0.0000 (-ADHD) | -0.0013 (-ADHD) | +0.0021 (+ADHD) |
| DefaultA | -0.0002 (-ADHD) | -0.0003 (-ADHD) | -0.0010 (-ADHD) | +0.0058 (+ADHD) |
| DefaultB | +0.0001 (+ADHD) | -0.0002 (-ADHD) | -0.0009 (-ADHD) | +0.0029 (+ADHD) |
| DefaultC | -0.0002 (-ADHD) | -0.0000 (-ADHD) | -0.0010 (-ADHD) | +0.0006 (+ADHD) |
| DorsAttnA | -0.0003 (-ADHD) | -0.0003 (-ADHD) | -0.0007 (-ADHD) | +0.0037 (+ADHD) |
| DorsAttnB | -0.0003 (-ADHD) | -0.0001 (-ADHD) | -0.0017 (-ADHD) | -0.0007 (-ADHD) |
| Limbic_OFC | +0.0007 (+ADHD) | +0.0001 (+ADHD) | +0.0009 (+ADHD) | +0.0067 (+ADHD) |
| Limbic_TempPole | +0.0001 (+ADHD) | -0.0000 (-ADHD) | -0.0002 (-ADHD) | +0.0431 (+ADHD) |
| SalVentAttnA | +0.0004 (+ADHD) | -0.0000 (-ADHD) | -0.0014 (-ADHD) | +0.0004 (+ADHD) |
| SalVentAttnB | +0.0005 (+ADHD) | -0.0000 (-ADHD) | -0.0014 (-ADHD) | +0.0019 (+ADHD) |
| SomMotA | -0.0004 (-ADHD) | -0.0002 (-ADHD) | -0.0018 (-ADHD) | -0.0004 (-ADHD) |
| SomMotB | +0.0002 (+ADHD) | -0.0002 (-ADHD) | -0.0015 (-ADHD) | -0.0000 (-ADHD) |
| TempPar | +0.0000 (+ADHD) | +0.0000 (+ADHD) | -0.0010 (-ADHD) | +0.0039 (+ADHD) |
| VisCent | -0.0005 (-ADHD) | -0.0001 (-ADHD) | -0.0000 (-ADHD) | +0.0026 (+ADHD) |
| VisPeri | -0.0001 (-ADHD) | +0.0001 (+ADHD) | -0.0003 (-ADHD) | +0.0026 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
