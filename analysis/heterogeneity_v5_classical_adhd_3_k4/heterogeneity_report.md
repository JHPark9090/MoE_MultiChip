# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_3 | **K**: 4 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.3952

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 139 | Executive | 0.249±0.001 | 0.252±0.000 | 0.248±0.001 | 0.252±0.001 |
| 1 | 6 | DMN | 0.255±0.002 | 0.252±0.001 | 0.250±0.001 | 0.243±0.003 |
| 2 | 142 | Executive | 0.250±0.001 | 0.252±0.000 | 0.248±0.000 | 0.250±0.001 |
| 3 | 1 | DMN | 0.259±0.000 | 0.248±0.000 | 0.241±0.000 | 0.252±0.000 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 196.811 | 0.0000 | Yes |
| Executive | 57.128 | 0.0000 | Yes |
| Salience | 100.176 | 0.0000 | Yes |
| SensoriMotor | 268.537 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3105

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 97 | DMN | 3.821 | 3.374 | 3.493 | 2.359 |
| 1 | 11 | DMN | 3.435 | 3.424 | 3.157 | 2.523 |
| 2 | 162 | DMN | 3.882 | 3.394 | 3.655 | 2.457 |
| 3 | 18 | DMN | 3.563 | 3.249 | 3.199 | 2.274 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1629

### Cluster 0 — Top 10 ROIs by Importance (N=125)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5464 | +0.0000 | +ADHD |
| 2 | 15 | V7 | VisCent | 0.5459 | -0.0000 | -ADHD |
| 3 | 157 | V3CD | VisCent | 0.5365 | -0.0002 | -ADHD |
| 4 | 20 | LO2 | VisCent | 0.5282 | -0.0003 | -ADHD |
| 5 | 176 | TE1m | TempPar | 0.5133 | +0.0004 | +ADHD |
| 6 | 65 | 47m | TempPar | 0.5044 | +0.0001 | +ADHD |
| 7 | 3 | V2 | VisPeri | 0.4976 | +0.0001 | +ADHD |
| 8 | 175 | STSva | TempPar | 0.4941 | -0.0000 | -ADHD |
| 9 | 126 | PHA3 | DefaultB | 0.4918 | -0.0006 | -ADHD |
| 10 | 19 | LO1 | VisCent | 0.4888 | +0.0002 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0011 |
| 120 | ProS | VisPeri | +0.0007 |
| 86 | 9a | TempPar | +0.0006 |
| 91 | 13l | Limbic_OFC | +0.0006 |
| 60 | a24 | DefaultC | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0007 |
| 126 | PHA3 | DefaultB | -0.0006 |
| 21 | PIT | VisCent | -0.0005 |
| 35 | 5m | SomMotA | -0.0004 |
| 28 | 7Pm | ContA | -0.0004 |

### Cluster 1 — Top 10 ROIs by Importance (N=8)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 163 | 25 | Limbic_OFC | 1.6473 | +0.0065 | +ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.5584 | +0.0032 | +ADHD |
| 3 | 165 | pOFC | Limbic_OFC | 1.5523 | +0.0116 | +ADHD |
| 4 | 89 | 10pp | Limbic_OFC | 1.5193 | +0.0047 | +ADHD |
| 5 | 177 | PI | SalVentAttnA | 1.5170 | +0.0194 | +ADHD |
| 6 | 164 | s32 | Limbic_OFC | 1.5119 | +0.0021 | +ADHD |
| 7 | 87 | 10v | Limbic_OFC | 1.4869 | +0.0030 | +ADHD |
| 8 | 60 | a24 | DefaultC | 1.4633 | +0.0050 | +ADHD |
| 9 | 91 | 13l | Limbic_OFC | 1.4615 | +0.0151 | +ADHD |
| 10 | 35 | 5m | SomMotA | 1.4381 | +0.0012 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.0755 |
| 171 | TGv | Limbic_TempPole | +0.0672 |
| 130 | TGd | Limbic_TempPole | +0.0493 |
| 134 | TF | Limbic_TempPole | +0.0492 |
| 109 | Pir | Limbic_TempPole | +0.0303 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0031 |
| 69 | 8BL | TempPar | -0.0030 |
| 101 | OP2-3 | SomMotB | -0.0026 |
| 114 | FOP2 | SomMotB | -0.0022 |
| 99 | OP4 | SomMotB | -0.0020 |

### Cluster 2 — Top 10 ROIs by Importance (N=47)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 35 | 5m | SomMotA | 0.9160 | -0.0023 | -ADHD |
| 2 | 52 | 3a | SomMotA | 0.8915 | -0.0015 | -ADHD |
| 3 | 92 | OFC | Limbic_OFC | 0.8668 | +0.0010 | +ADHD |
| 4 | 70 | 9p | TempPar | 0.8597 | -0.0013 | -ADHD |
| 5 | 31 | 23d | DefaultC | 0.8559 | -0.0011 | -ADHD |
| 6 | 163 | 25 | Limbic_OFC | 0.8522 | +0.0018 | +ADHD |
| 7 | 37 | 23c | SalVentAttnA | 0.8517 | -0.0012 | -ADHD |
| 8 | 166 | PoI1 | SalVentAttnA | 0.8450 | -0.0012 | -ADHD |
| 9 | 101 | OP2-3 | SomMotB | 0.8436 | -0.0003 | -ADHD |
| 10 | 7 | 4 | SomMotA | 0.8411 | -0.0014 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 91 | 13l | Limbic_OFC | +0.0028 |
| 165 | pOFC | Limbic_OFC | +0.0021 |
| 163 | 25 | Limbic_OFC | +0.0018 |
| 130 | TGd | Limbic_TempPole | +0.0015 |
| 88 | a10p | ContC | +0.0014 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 35 | 5m | SomMotA | -0.0023 |
| 143 | IP2 | ContB | -0.0020 |
| 8 | 3b | SomMotA | -0.0019 |
| 173 | LBelt | SomMotB | -0.0019 |
| 25 | SFL | TempPar | -0.0018 |

### Cluster 3 — Top 10 ROIs by Importance (N=108)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.6510 | -0.0001 | -ADHD |
| 2 | 124 | A5 | DefaultA | 0.6189 | -0.0007 | -ADHD |
| 3 | 103 | RI | SomMotB | 0.6159 | -0.0005 | -ADHD |
| 4 | 175 | STSva | TempPar | 0.6148 | -0.0008 | -ADHD |
| 5 | 73 | 44 | TempPar | 0.6096 | -0.0003 | -ADHD |
| 6 | 101 | OP2-3 | SomMotB | 0.6052 | +0.0005 | +ADHD |
| 7 | 126 | PHA3 | DefaultB | 0.6052 | -0.0005 | -ADHD |
| 8 | 52 | 3a | SomMotA | 0.5999 | -0.0011 | -ADHD |
| 9 | 65 | 47m | TempPar | 0.5995 | -0.0002 | -ADHD |
| 10 | 98 | 43 | SalVentAttnA | 0.5994 | +0.0002 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0009 |
| 177 | PI | SalVentAttnA | +0.0006 |
| 91 | 13l | Limbic_OFC | +0.0006 |
| 101 | OP2-3 | SomMotB | +0.0005 |
| 166 | PoI1 | SalVentAttnA | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0012 |
| 52 | 3a | SomMotA | -0.0011 |
| 117 | EC | Limbic_TempPole | -0.0010 |
| 121 | PeEc | Limbic_TempPole | -0.0010 |
| 142 | PGp | DorsAttnA | -0.0010 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.3861 | 1.1429 | 0.6900 | 0.4921 |
| ContB | 0.3940 | 0.8827 | 0.6689 | 0.5107 |
| ContC | 0.3738 | 0.8647 | 0.5971 | 0.4722 |
| DefaultA | 0.4421 | 0.9535 | 0.6938 | 0.5889 |
| DefaultB | 0.4404 | 0.9159 | 0.6834 | 0.5564 |
| DefaultC | 0.4318 | 1.1536 | 0.7680 | 0.5513 |
| DorsAttnA | 0.4142 | 0.9376 | 0.6262 | 0.5014 |
| DorsAttnB | 0.3816 | 0.9957 | 0.6983 | 0.4940 |
| Limbic_OFC | 0.3846 | 1.5165 | 0.7817 | 0.4967 |
| Limbic_TempPole | 0.3947 | 1.2053 | 0.6686 | 0.5203 |
| SalVentAttnA | 0.3977 | 1.0464 | 0.7357 | 0.5252 |
| SalVentAttnB | 0.4112 | 1.0146 | 0.7431 | 0.5398 |
| SomMotA | 0.3963 | 1.2500 | 0.8238 | 0.5518 |
| SomMotB | 0.4107 | 0.9707 | 0.7180 | 0.5521 |
| TempPar | 0.4342 | 0.9907 | 0.7236 | 0.5594 |
| VisCent | 0.4827 | 0.8200 | 0.6140 | 0.5506 |
| VisPeri | 0.4534 | 0.8307 | 0.6179 | 0.5275 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | -0.0002 (-ADHD) | +0.0015 (+ADHD) | -0.0010 (-ADHD) | -0.0003 (-ADHD) |
| ContB | -0.0001 (-ADHD) | +0.0001 (+ADHD) | -0.0009 (-ADHD) | -0.0004 (-ADHD) |
| ContC | +0.0001 (+ADHD) | +0.0022 (+ADHD) | -0.0006 (-ADHD) | -0.0005 (-ADHD) |
| DefaultA | -0.0002 (-ADHD) | +0.0060 (+ADHD) | -0.0009 (-ADHD) | -0.0005 (-ADHD) |
| DefaultB | -0.0000 (-ADHD) | +0.0032 (+ADHD) | -0.0002 (-ADHD) | -0.0004 (-ADHD) |
| DefaultC | +0.0001 (+ADHD) | +0.0012 (+ADHD) | -0.0009 (-ADHD) | -0.0004 (-ADHD) |
| DorsAttnA | -0.0001 (-ADHD) | +0.0034 (+ADHD) | -0.0006 (-ADHD) | -0.0005 (-ADHD) |
| DorsAttnB | -0.0000 (-ADHD) | -0.0002 (-ADHD) | -0.0014 (-ADHD) | -0.0003 (-ADHD) |
| Limbic_OFC | +0.0002 (+ADHD) | +0.0067 (+ADHD) | +0.0015 (+ADHD) | +0.0003 (+ADHD) |
| Limbic_TempPole | +0.0002 (+ADHD) | +0.0396 (+ADHD) | +0.0004 (+ADHD) | -0.0004 (-ADHD) |
| SalVentAttnA | +0.0001 (+ADHD) | +0.0007 (+ADHD) | -0.0009 (-ADHD) | +0.0001 (+ADHD) |
| SalVentAttnB | +0.0001 (+ADHD) | +0.0020 (+ADHD) | -0.0007 (-ADHD) | +0.0001 (+ADHD) |
| SomMotA | -0.0001 (-ADHD) | +0.0005 (+ADHD) | -0.0016 (-ADHD) | -0.0006 (-ADHD) |
| SomMotB | +0.0000 (+ADHD) | +0.0003 (+ADHD) | -0.0010 (-ADHD) | -0.0002 (-ADHD) |
| TempPar | +0.0001 (+ADHD) | +0.0041 (+ADHD) | -0.0005 (-ADHD) | -0.0003 (-ADHD) |
| VisCent | -0.0001 (-ADHD) | +0.0027 (+ADHD) | -0.0001 (-ADHD) | -0.0006 (-ADHD) |
| VisPeri | +0.0002 (+ADHD) | +0.0026 (+ADHD) | +0.0002 (+ADHD) | -0.0003 (-ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
