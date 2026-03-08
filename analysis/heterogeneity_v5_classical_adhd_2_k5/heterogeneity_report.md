# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_2 | **K**: 5 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.5197

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 76 | Internal | 0.542±0.025 | 0.458±0.025 |
| 1 | 64 | External | 0.421±0.024 | 0.579±0.024 |
| 2 | 9 | Internal | 0.729±0.061 | 0.271±0.061 |
| 3 | 11 | External | 0.272±0.063 | 0.728±0.063 |
| 4 | 128 | External | 0.479±0.017 | 0.521±0.017 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 546.056 | 0.0000 | Yes |
| External | 546.056 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3324

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 92 | Internal | 4.047 | 1.443 |
| 1 | 22 | Internal | 5.506 | 4.621 |
| 2 | 30 | Internal | 4.070 | 3.174 |
| 3 | 70 | Internal | 3.794 | 1.173 |
| 4 | 74 | Internal | 6.496 | 1.301 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1065

### Cluster 0 — Top 10 ROIs by Importance (N=106)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 20 | LO2 | VisCent | 0.7638 | -0.0002 | -ADHD |
| 2 | 171 | TGv | Limbic_TempPole | 0.7490 | +0.0006 | +ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.7338 | -0.0002 | -ADHD |
| 4 | 38 | 5L | DorsAttnB | 0.7188 | -0.0002 | -ADHD |
| 5 | 160 | 31pd | DefaultC | 0.7123 | -0.0001 | -ADHD |
| 6 | 168 | FOP5 | SalVentAttnB | 0.6986 | +0.0008 | +ADHD |
| 7 | 73 | 44 | TempPar | 0.6977 | +0.0004 | +ADHD |
| 8 | 174 | A4 | SomMotB | 0.6953 | -0.0003 | -ADHD |
| 9 | 135 | TE2p | Limbic_TempPole | 0.6645 | -0.0004 | -ADHD |
| 10 | 149 | PGi | DefaultC | 0.6431 | -0.0001 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0009 |
| 13 | RSC | ContA | +0.0008 |
| 168 | FOP5 | SalVentAttnB | +0.0008 |
| 32 | v23ab | DefaultC | +0.0007 |
| 166 | PoI1 | SalVentAttnA | +0.0007 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 45 | 7Pl | DorsAttnA | -0.0008 |
| 152 | VMV1 | VisPeri | -0.0008 |
| 162 | VVC | VisCent | -0.0008 |
| 117 | EC | Limbic_TempPole | -0.0008 |
| 22 | MT | DorsAttnA | -0.0008 |

### Cluster 1 — Top 10 ROIs by Importance (N=28)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.3905 | +0.0014 | +ADHD |
| 2 | 38 | 5L | DorsAttnB | 1.3405 | -0.0012 | -ADHD |
| 3 | 54 | 6mp | SomMotA | 1.2032 | -0.0008 | -ADHD |
| 4 | 161 | 31a | ContA | 1.1954 | -0.0000 | -ADHD |
| 5 | 134 | TF | Limbic_TempPole | 1.1547 | -0.0009 | -ADHD |
| 6 | 60 | a24 | DefaultC | 1.1310 | +0.0014 | +ADHD |
| 7 | 165 | pOFC | Limbic_OFC | 1.1023 | +0.0023 | +ADHD |
| 8 | 177 | PI | SalVentAttnA | 1.0744 | -0.0001 | -ADHD |
| 9 | 169 | p10p | ContC | 1.0712 | +0.0014 | +ADHD |
| 10 | 160 | 31pd | DefaultC | 1.0692 | -0.0010 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 88 | a10p | ContC | +0.0036 |
| 91 | 13l | Limbic_OFC | +0.0033 |
| 165 | pOFC | Limbic_OFC | +0.0023 |
| 89 | 10pp | Limbic_OFC | +0.0020 |
| 90 | 11l | Limbic_OFC | +0.0018 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 173 | LBelt | SomMotB | -0.0019 |
| 135 | TE2p | Limbic_TempPole | -0.0018 |
| 24 | PSL | SalVentAttnA | -0.0016 |
| 172 | MBelt | SomMotB | -0.0016 |
| 103 | RI | SomMotB | -0.0014 |

### Cluster 2 — Top 10 ROIs by Importance (N=61)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 38 | 5L | DorsAttnB | 0.9551 | -0.0019 | -ADHD |
| 2 | 171 | TGv | Limbic_TempPole | 0.9458 | +0.0008 | +ADHD |
| 3 | 168 | FOP5 | SalVentAttnB | 0.8752 | -0.0007 | -ADHD |
| 4 | 73 | 44 | TempPar | 0.8722 | -0.0003 | -ADHD |
| 5 | 134 | TF | Limbic_TempPole | 0.8606 | -0.0018 | -ADHD |
| 6 | 161 | 31a | ContA | 0.8496 | -0.0023 | -ADHD |
| 7 | 169 | p10p | ContC | 0.8433 | -0.0010 | -ADHD |
| 8 | 160 | 31pd | DefaultC | 0.8335 | -0.0019 | -ADHD |
| 9 | 54 | 6mp | SomMotA | 0.8303 | -0.0011 | -ADHD |
| 10 | 66 | 8Av | ContC | 0.8040 | -0.0016 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0026 |
| 165 | pOFC | Limbic_OFC | +0.0011 |
| 89 | 10pp | Limbic_OFC | +0.0010 |
| 163 | 25 | Limbic_OFC | +0.0010 |
| 90 | 11l | Limbic_OFC | +0.0010 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 20 | LO2 | VisCent | -0.0023 |
| 161 | 31a | ContA | -0.0023 |
| 38 | 5L | DorsAttnB | -0.0019 |
| 160 | 31pd | DefaultC | -0.0019 |
| 76 | a47r | ContC | -0.0019 |

### Cluster 3 — Top 10 ROIs by Importance (N=5)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 171 | TGv | Limbic_TempPole | 2.4134 | +0.1774 | +ADHD |
| 2 | 177 | PI | SalVentAttnA | 2.0901 | +0.0357 | +ADHD |
| 3 | 165 | pOFC | Limbic_OFC | 1.9846 | +0.0215 | +ADHD |
| 4 | 38 | 5L | DorsAttnB | 1.9622 | -0.0001 | -ADHD |
| 5 | 60 | a24 | DefaultC | 1.9436 | +0.0060 | +ADHD |
| 6 | 92 | OFC | Limbic_OFC | 1.8934 | +0.0058 | +ADHD |
| 7 | 161 | 31a | ContA | 1.7916 | +0.0016 | +ADHD |
| 8 | 89 | 10pp | Limbic_OFC | 1.7777 | +0.0044 | +ADHD |
| 9 | 90 | 11l | Limbic_OFC | 1.7357 | +0.0132 | +ADHD |
| 10 | 54 | 6mp | SomMotA | 1.7262 | -0.0017 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.1774 |
| 121 | PeEc | Limbic_TempPole | +0.1071 |
| 134 | TF | Limbic_TempPole | +0.1058 |
| 130 | TGd | Limbic_TempPole | +0.0721 |
| 117 | EC | Limbic_TempPole | +0.0576 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 114 | FOP2 | SomMotB | -0.0056 |
| 69 | 8BL | TempPar | -0.0054 |
| 102 | 52 | SomMotB | -0.0052 |
| 101 | OP2-3 | SomMotB | -0.0051 |
| 42 | SCEF | SalVentAttnA | -0.0041 |

### Cluster 4 — Top 10 ROIs by Importance (N=88)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 20 | LO2 | VisCent | 0.6657 | -0.0004 | -ADHD |
| 2 | 160 | 31pd | DefaultC | 0.6116 | +0.0001 | +ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.6102 | -0.0000 | -ADHD |
| 4 | 171 | TGv | Limbic_TempPole | 0.6001 | +0.0004 | +ADHD |
| 5 | 168 | FOP5 | SalVentAttnB | 0.5745 | -0.0001 | -ADHD |
| 6 | 73 | 44 | TempPar | 0.5634 | -0.0008 | -ADHD |
| 7 | 169 | p10p | ContC | 0.5621 | +0.0007 | +ADHD |
| 8 | 32 | v23ab | DefaultC | 0.5557 | -0.0004 | -ADHD |
| 9 | 18 | V3B | VisCent | 0.5509 | -0.0007 | -ADHD |
| 10 | 76 | a47r | ContC | 0.5376 | +0.0002 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0010 |
| 177 | PI | SalVentAttnA | +0.0008 |
| 169 | p10p | ContC | +0.0007 |
| 121 | PeEc | Limbic_TempPole | +0.0007 |
| 91 | 13l | Limbic_OFC | +0.0007 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 138 | TPOJ1 | DefaultA | -0.0008 |
| 38 | 5L | DorsAttnB | -0.0008 |
| 73 | 44 | TempPar | -0.0008 |
| 174 | A4 | SomMotB | -0.0008 |
| 122 | STGa | DefaultA | -0.0007 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.5281 | 0.9110 | 0.6562 | 1.4324 | 0.4235 |
| ContB | 0.4652 | 0.7208 | 0.5755 | 0.9668 | 0.3791 |
| ContC | 0.4971 | 0.7452 | 0.6197 | 1.0798 | 0.4215 |
| DefaultA | 0.5179 | 0.6862 | 0.6109 | 1.0241 | 0.3998 |
| DefaultB | 0.5170 | 0.7204 | 0.6187 | 0.9508 | 0.4240 |
| DefaultC | 0.4996 | 0.8473 | 0.6233 | 1.2465 | 0.4124 |
| DorsAttnA | 0.4687 | 0.6809 | 0.5428 | 1.0385 | 0.3971 |
| DorsAttnB | 0.4777 | 0.8180 | 0.6123 | 1.1352 | 0.3766 |
| Limbic_OFC | 0.4545 | 0.9630 | 0.5996 | 1.6183 | 0.3597 |
| Limbic_TempPole | 0.5593 | 0.8780 | 0.6922 | 1.5377 | 0.4428 |
| SalVentAttnA | 0.4744 | 0.7948 | 0.5930 | 1.0923 | 0.3707 |
| SalVentAttnB | 0.4863 | 0.7957 | 0.6228 | 1.0850 | 0.3969 |
| SomMotA | 0.4686 | 0.8692 | 0.6249 | 1.2540 | 0.3446 |
| SomMotB | 0.5266 | 0.7868 | 0.6359 | 1.0528 | 0.4001 |
| TempPar | 0.4851 | 0.7460 | 0.5960 | 1.0240 | 0.4014 |
| VisCent | 0.5106 | 0.6220 | 0.5467 | 0.8560 | 0.4499 |
| VisPeri | 0.4955 | 0.6447 | 0.5458 | 0.8868 | 0.4340 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | +0.0000 (+ADHD) | -0.0006 (-ADHD) | -0.0011 (-ADHD) | +0.0013 (+ADHD) | -0.0003 (-ADHD) |
| ContB | -0.0001 (-ADHD) | -0.0008 (-ADHD) | -0.0007 (-ADHD) | -0.0008 (-ADHD) | -0.0003 (-ADHD) |
| ContC | -0.0001 (-ADHD) | +0.0001 (+ADHD) | -0.0012 (-ADHD) | +0.0025 (+ADHD) | +0.0001 (+ADHD) |
| DefaultA | -0.0000 (-ADHD) | -0.0007 (-ADHD) | -0.0008 (-ADHD) | +0.0081 (+ADHD) | -0.0005 (-ADHD) |
| DefaultB | -0.0002 (-ADHD) | -0.0007 (-ADHD) | +0.0001 (+ADHD) | +0.0039 (+ADHD) | -0.0001 (-ADHD) |
| DefaultC | +0.0001 (+ADHD) | -0.0005 (-ADHD) | -0.0007 (-ADHD) | +0.0009 (+ADHD) | -0.0001 (-ADHD) |
| DorsAttnA | -0.0004 (-ADHD) | -0.0006 (-ADHD) | -0.0002 (-ADHD) | +0.0049 (+ADHD) | -0.0002 (-ADHD) |
| DorsAttnB | +0.0001 (+ADHD) | -0.0008 (-ADHD) | -0.0011 (-ADHD) | -0.0009 (-ADHD) | -0.0003 (-ADHD) |
| Limbic_OFC | -0.0000 (-ADHD) | +0.0017 (+ADHD) | +0.0011 (+ADHD) | +0.0097 (+ADHD) | +0.0003 (+ADHD) |
| Limbic_TempPole | -0.0002 (-ADHD) | -0.0001 (-ADHD) | -0.0000 (-ADHD) | +0.0746 (+ADHD) | +0.0002 (+ADHD) |
| SalVentAttnA | +0.0002 (+ADHD) | -0.0006 (-ADHD) | -0.0004 (-ADHD) | +0.0005 (+ADHD) | +0.0001 (+ADHD) |
| SalVentAttnB | +0.0003 (+ADHD) | -0.0004 (-ADHD) | -0.0005 (-ADHD) | +0.0022 (+ADHD) | +0.0001 (+ADHD) |
| SomMotA | -0.0002 (-ADHD) | -0.0006 (-ADHD) | -0.0012 (-ADHD) | -0.0006 (-ADHD) | -0.0003 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | -0.0010 (-ADHD) | -0.0005 (-ADHD) | -0.0006 (-ADHD) | -0.0000 (-ADHD) |
| TempPar | +0.0001 (+ADHD) | -0.0002 (-ADHD) | -0.0006 (-ADHD) | +0.0052 (+ADHD) | -0.0000 (-ADHD) |
| VisCent | -0.0003 (-ADHD) | +0.0001 (+ADHD) | -0.0006 (-ADHD) | +0.0032 (+ADHD) | -0.0002 (-ADHD) |
| VisPeri | -0.0002 (-ADHD) | -0.0001 (-ADHD) | +0.0002 (+ADHD) | +0.0035 (+ADHD) | +0.0001 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
