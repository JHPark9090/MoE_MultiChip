# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_2 | **K**: 5 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.5330

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 99 | Internal | 0.506±0.005 | 0.494±0.005 |
| 1 | 6 | Internal | 0.631±0.022 | 0.369±0.022 |
| 2 | 43 | External | 0.474±0.013 | 0.526±0.013 |
| 3 | 10 | Internal | 0.539±0.018 | 0.461±0.018 |
| 4 | 130 | External | 0.492±0.004 | 0.508±0.004 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 587.312 | 0.0000 | Yes |
| External | 587.311 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.1561

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 41 | External | 1.270 | 1.754 |
| 1 | 53 | Internal | 3.268 | 1.537 |
| 2 | 49 | External | 1.480 | 2.099 |
| 3 | 75 | Internal | 2.201 | 1.372 |
| 4 | 70 | Internal | 1.382 | 1.284 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1353

### Cluster 0 — Top 10 ROIs by Importance (N=117)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 157 | V3CD | VisCent | 0.4537 | -0.0003 | -ADHD |
| 2 | 18 | V3B | VisCent | 0.4458 | -0.0004 | -ADHD |
| 3 | 76 | a47r | ContC | 0.4377 | -0.0000 | -ADHD |
| 4 | 128 | STSdp | DefaultA | 0.4323 | -0.0002 | -ADHD |
| 5 | 131 | TE1a | TempPar | 0.4247 | -0.0000 | -ADHD |
| 6 | 21 | PIT | VisCent | 0.4196 | -0.0005 | -ADHD |
| 7 | 16 | IPS1 | DorsAttnA | 0.4193 | -0.0003 | -ADHD |
| 8 | 100 | OP1 | SomMotB | 0.4094 | +0.0003 | +ADHD |
| 9 | 113 | FOP3 | SalVentAttnA | 0.4080 | +0.0001 | +ADHD |
| 10 | 20 | LO2 | VisCent | 0.4065 | -0.0001 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0008 |
| 120 | ProS | VisPeri | +0.0005 |
| 91 | 13l | Limbic_OFC | +0.0004 |
| 158 | LO3 | DorsAttnA | +0.0004 |
| 169 | p10p | ContC | +0.0003 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0007 |
| 50 | 1 | SomMotA | -0.0005 |
| 174 | A4 | SomMotB | -0.0005 |
| 21 | PIT | VisCent | -0.0005 |
| 94 | LIPd | ContB | -0.0004 |

### Cluster 1 — Top 10 ROIs by Importance (N=97)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 128 | STSdp | DefaultA | 0.5536 | -0.0002 | -ADHD |
| 2 | 174 | A4 | SomMotB | 0.5350 | -0.0002 | -ADHD |
| 3 | 131 | TE1a | TempPar | 0.5279 | +0.0004 | +ADHD |
| 4 | 100 | OP1 | SomMotB | 0.5266 | +0.0006 | +ADHD |
| 5 | 1 | MST | DorsAttnA | 0.5130 | -0.0002 | -ADHD |
| 6 | 17 | FFC | DorsAttnA | 0.5033 | -0.0005 | -ADHD |
| 7 | 135 | TE2p | Limbic_TempPole | 0.4978 | -0.0004 | -ADHD |
| 8 | 81 | IFSa | ContB | 0.4975 | +0.0000 | +ADHD |
| 9 | 10 | PEF | ContB | 0.4963 | -0.0002 | -ADHD |
| 10 | 16 | IPS1 | DorsAttnA | 0.4928 | -0.0003 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0009 |
| 111 | AAIC | SalVentAttnB | +0.0007 |
| 166 | PoI1 | SalVentAttnA | +0.0006 |
| 13 | RSC | ContA | +0.0006 |
| 100 | OP1 | SomMotB | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0010 |
| 22 | MT | DorsAttnA | -0.0009 |
| 142 | PGp | DorsAttnA | -0.0009 |
| 136 | PHT | ContB | -0.0009 |
| 121 | PeEc | Limbic_TempPole | -0.0008 |

### Cluster 2 — Top 10 ROIs by Importance (N=26)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.1039 | +0.0013 | +ADHD |
| 2 | 164 | s32 | Limbic_OFC | 0.9241 | +0.0008 | +ADHD |
| 3 | 54 | 6mp | SomMotA | 0.8987 | -0.0006 | -ADHD |
| 4 | 100 | OP1 | SomMotB | 0.8619 | -0.0009 | -ADHD |
| 5 | 41 | 7AL | DorsAttnB | 0.8619 | -0.0006 | -ADHD |
| 6 | 177 | PI | SalVentAttnA | 0.8492 | -0.0001 | -ADHD |
| 7 | 96 | i6-8 | ContC | 0.8374 | -0.0008 | -ADHD |
| 8 | 163 | 25 | Limbic_OFC | 0.8234 | +0.0009 | +ADHD |
| 9 | 161 | 31a | ContA | 0.8204 | -0.0001 | -ADHD |
| 10 | 7 | 4 | SomMotA | 0.8203 | -0.0005 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 88 | a10p | ContC | +0.0032 |
| 91 | 13l | Limbic_OFC | +0.0027 |
| 93 | 47s | TempPar | +0.0015 |
| 165 | pOFC | Limbic_OFC | +0.0015 |
| 92 | OFC | Limbic_OFC | +0.0013 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 135 | TE2p | Limbic_TempPole | -0.0015 |
| 173 | LBelt | SomMotB | -0.0015 |
| 144 | IP1 | ContB | -0.0013 |
| 29 | 7m | DefaultC | -0.0013 |
| 24 | PSL | SalVentAttnA | -0.0013 |

### Cluster 3 — Top 10 ROIs by Importance (N=43)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 100 | OP1 | SomMotB | 0.6659 | -0.0001 | -ADHD |
| 2 | 54 | 6mp | SomMotA | 0.6659 | -0.0012 | -ADHD |
| 3 | 41 | 7AL | DorsAttnB | 0.6462 | -0.0021 | -ADHD |
| 4 | 96 | i6-8 | ContC | 0.6357 | -0.0015 | -ADHD |
| 5 | 10 | PEF | ContB | 0.6331 | +0.0001 | +ADHD |
| 6 | 128 | STSdp | DefaultA | 0.6317 | -0.0008 | -ADHD |
| 7 | 23 | A1 | SomMotB | 0.6230 | +0.0005 | +ADHD |
| 8 | 62 | 8BM | ContC | 0.6203 | -0.0012 | -ADHD |
| 9 | 66 | 8Av | ContC | 0.6154 | -0.0010 | -ADHD |
| 10 | 161 | 31a | ContA | 0.6154 | -0.0018 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0027 |
| 89 | 10pp | Limbic_OFC | +0.0011 |
| 164 | s32 | Limbic_OFC | +0.0011 |
| 154 | PHA2 | DefaultB | +0.0010 |
| 133 | TE2a | Limbic_TempPole | +0.0010 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 41 | 7AL | DorsAttnB | -0.0021 |
| 46 | 7PC | DorsAttnB | -0.0019 |
| 161 | 31a | ContA | -0.0018 |
| 51 | 2 | SomMotA | -0.0017 |
| 38 | 5L | DorsAttnB | -0.0016 |

### Cluster 4 — Top 10 ROIs by Importance (N=5)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 177 | PI | SalVentAttnA | 1.6521 | +0.0282 | +ADHD |
| 2 | 163 | 25 | Limbic_OFC | 1.5709 | +0.0082 | +ADHD |
| 3 | 92 | OFC | Limbic_OFC | 1.5047 | +0.0046 | +ADHD |
| 4 | 106 | TA2 | SomMotB | 1.3877 | +0.0213 | +ADHD |
| 5 | 164 | s32 | Limbic_OFC | 1.3538 | +0.0031 | +ADHD |
| 6 | 165 | pOFC | Limbic_OFC | 1.3357 | +0.0144 | +ADHD |
| 7 | 121 | PeEc | Limbic_TempPole | 1.3164 | +0.1017 | +ADHD |
| 8 | 130 | TGd | Limbic_TempPole | 1.3159 | +0.0642 | +ADHD |
| 9 | 54 | 6mp | SomMotA | 1.2686 | -0.0012 | -ADHD |
| 10 | 144 | IP1 | ContB | 1.2320 | +0.0017 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.1017 |
| 171 | TGv | Limbic_TempPole | +0.0859 |
| 130 | TGd | Limbic_TempPole | +0.0642 |
| 134 | TF | Limbic_TempPole | +0.0538 |
| 117 | EC | Limbic_TempPole | +0.0417 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 101 | OP2-3 | SomMotB | -0.0049 |
| 102 | 52 | SomMotB | -0.0049 |
| 114 | FOP2 | SomMotB | -0.0044 |
| 69 | 8BL | TempPar | -0.0041 |
| 62 | 8BM | ContC | -0.0036 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.3384 | 0.4261 | 0.7023 | 0.5257 | 1.0970 |
| ContB | 0.3566 | 0.4479 | 0.6608 | 0.5513 | 0.8692 |
| ContC | 0.3602 | 0.4428 | 0.6339 | 0.5375 | 0.9041 |
| DefaultA | 0.3446 | 0.4572 | 0.5696 | 0.5129 | 0.8476 |
| DefaultB | 0.3398 | 0.4212 | 0.5615 | 0.4954 | 0.7442 |
| DefaultC | 0.3084 | 0.3837 | 0.6233 | 0.4731 | 0.9022 |
| DorsAttnA | 0.3853 | 0.4613 | 0.6506 | 0.5121 | 0.9812 |
| DorsAttnB | 0.3526 | 0.4402 | 0.7280 | 0.5706 | 0.9966 |
| Limbic_OFC | 0.3010 | 0.3822 | 0.7775 | 0.4963 | 1.2864 |
| Limbic_TempPole | 0.3275 | 0.4202 | 0.6247 | 0.5066 | 1.1065 |
| SalVentAttnA | 0.3164 | 0.4066 | 0.6598 | 0.5177 | 0.8954 |
| SalVentAttnB | 0.3306 | 0.4219 | 0.6601 | 0.5367 | 0.8854 |
| SomMotA | 0.3106 | 0.4156 | 0.7425 | 0.5685 | 1.0552 |
| SomMotB | 0.3501 | 0.4604 | 0.6705 | 0.5576 | 0.8929 |
| TempPar | 0.3274 | 0.4109 | 0.6004 | 0.4926 | 0.8156 |
| VisCent | 0.4003 | 0.4511 | 0.5398 | 0.4725 | 0.7392 |
| VisPeri | 0.3479 | 0.3969 | 0.5051 | 0.4367 | 0.6908 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | -0.0003 (-ADHD) | -0.0000 (-ADHD) | -0.0006 (-ADHD) | -0.0008 (-ADHD) | +0.0009 (+ADHD) |
| ContB | -0.0002 (-ADHD) | -0.0002 (-ADHD) | -0.0009 (-ADHD) | -0.0005 (-ADHD) | -0.0008 (-ADHD) |
| ContC | +0.0000 (+ADHD) | -0.0003 (-ADHD) | -0.0001 (-ADHD) | -0.0009 (-ADHD) | +0.0018 (+ADHD) |
| DefaultA | -0.0003 (-ADHD) | -0.0002 (-ADHD) | -0.0007 (-ADHD) | -0.0007 (-ADHD) | +0.0071 (+ADHD) |
| DefaultB | -0.0001 (-ADHD) | -0.0003 (-ADHD) | -0.0006 (-ADHD) | +0.0003 (+ADHD) | +0.0030 (+ADHD) |
| DefaultC | +0.0000 (+ADHD) | -0.0001 (-ADHD) | -0.0005 (-ADHD) | -0.0005 (-ADHD) | +0.0006 (+ADHD) |
| DorsAttnA | -0.0002 (-ADHD) | -0.0005 (-ADHD) | -0.0007 (-ADHD) | -0.0001 (-ADHD) | +0.0045 (+ADHD) |
| DorsAttnB | -0.0002 (-ADHD) | +0.0000 (+ADHD) | -0.0007 (-ADHD) | -0.0013 (-ADHD) | -0.0009 (-ADHD) |
| Limbic_OFC | +0.0002 (+ADHD) | +0.0000 (+ADHD) | +0.0013 (+ADHD) | +0.0012 (+ADHD) | +0.0075 (+ADHD) |
| Limbic_TempPole | +0.0001 (+ADHD) | -0.0003 (-ADHD) | -0.0001 (-ADHD) | +0.0003 (+ADHD) | +0.0515 (+ADHD) |
| SalVentAttnA | +0.0000 (+ADHD) | +0.0003 (+ADHD) | -0.0006 (-ADHD) | -0.0005 (-ADHD) | +0.0004 (+ADHD) |
| SalVentAttnB | -0.0000 (-ADHD) | +0.0003 (+ADHD) | -0.0005 (-ADHD) | -0.0004 (-ADHD) | +0.0019 (+ADHD) |
| SomMotA | -0.0002 (-ADHD) | -0.0002 (-ADHD) | -0.0006 (-ADHD) | -0.0013 (-ADHD) | -0.0005 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | -0.0000 (-ADHD) | -0.0010 (-ADHD) | -0.0004 (-ADHD) | -0.0001 (-ADHD) |
| TempPar | +0.0000 (+ADHD) | -0.0000 (-ADHD) | -0.0003 (-ADHD) | -0.0003 (-ADHD) | +0.0045 (+ADHD) |
| VisCent | -0.0001 (-ADHD) | -0.0004 (-ADHD) | +0.0000 (+ADHD) | -0.0003 (-ADHD) | +0.0031 (+ADHD) |
| VisPeri | +0.0001 (+ADHD) | -0.0002 (-ADHD) | -0.0002 (-ADHD) | +0.0003 (+ADHD) | +0.0029 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
