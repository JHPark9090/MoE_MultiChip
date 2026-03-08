# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_3 | **K**: 5 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.4452

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 68 | Executive | 0.250±0.001 | 0.253±0.001 | 0.245±0.001 | 0.251±0.001 |
| 1 | 7 | Salience | 0.241±0.002 | 0.239±0.003 | 0.267±0.004 | 0.254±0.002 |
| 2 | 62 | SensoriMotor | 0.247±0.001 | 0.249±0.001 | 0.250±0.001 | 0.253±0.001 |
| 3 | 1 | Executive | 0.262±0.000 | 0.268±0.000 | 0.224±0.000 | 0.246±0.000 |
| 4 | 150 | SensoriMotor | 0.249±0.001 | 0.251±0.001 | 0.248±0.001 | 0.252±0.000 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 271.011 | 0.0000 | Yes |
| Executive | 407.591 | 0.0000 | Yes |
| Salience | 682.518 | 0.0000 | Yes |
| SensoriMotor | 65.992 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.1369

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 69 | Salience | 1.546 | 1.266 | 1.573 | 1.185 |
| 1 | 40 | DMN | 2.310 | 1.450 | 2.268 | 1.489 |
| 2 | 50 | Executive | 1.399 | 1.902 | 1.437 | 1.166 |
| 3 | 61 | Executive | 1.166 | 1.362 | 1.143 | 1.296 |
| 4 | 68 | Salience | 1.948 | 1.865 | 2.658 | 1.152 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1338

### Cluster 0 — Top 10 ROIs by Importance (N=26)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.3117 | +0.0015 | +ADHD |
| 2 | 164 | s32 | Limbic_OFC | 1.1427 | +0.0010 | +ADHD |
| 3 | 39 | 24dd | SomMotA | 1.1084 | +0.0002 | +ADHD |
| 4 | 165 | pOFC | Limbic_OFC | 1.0975 | +0.0021 | +ADHD |
| 5 | 163 | 25 | Limbic_OFC | 1.0873 | +0.0012 | +ADHD |
| 6 | 70 | 9p | TempPar | 1.0534 | -0.0016 | -ADHD |
| 7 | 40 | 24dv | SalVentAttnA | 1.0337 | -0.0006 | -ADHD |
| 8 | 54 | 6mp | SomMotA | 1.0226 | -0.0006 | -ADHD |
| 9 | 177 | PI | SalVentAttnA | 1.0156 | -0.0002 | -ADHD |
| 10 | 179 | p24 | DefaultC | 1.0131 | -0.0007 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 88 | a10p | ContC | +0.0043 |
| 91 | 13l | Limbic_OFC | +0.0037 |
| 165 | pOFC | Limbic_OFC | +0.0021 |
| 93 | 47s | TempPar | +0.0020 |
| 92 | OFC | Limbic_OFC | +0.0015 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 173 | LBelt | SomMotB | -0.0019 |
| 172 | MBelt | SomMotB | -0.0017 |
| 140 | TPOJ3 | DorsAttnA | -0.0017 |
| 29 | 7m | DefaultC | -0.0016 |
| 27 | STV | DefaultA | -0.0016 |

### Cluster 1 — Top 10 ROIs by Importance (N=105)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 134 | TF | Limbic_TempPole | 0.6487 | -0.0006 | -ADHD |
| 2 | 149 | PGi | DefaultC | 0.6415 | -0.0003 | -ADHD |
| 3 | 70 | 9p | TempPar | 0.6399 | +0.0002 | +ADHD |
| 4 | 128 | STSdp | DefaultA | 0.6378 | -0.0002 | -ADHD |
| 5 | 32 | v23ab | DefaultC | 0.6377 | +0.0003 | +ADHD |
| 6 | 80 | IFSp | ContB | 0.6291 | +0.0000 | +ADHD |
| 7 | 175 | STSva | TempPar | 0.6238 | -0.0005 | -ADHD |
| 8 | 23 | A1 | SomMotB | 0.6237 | -0.0005 | -ADHD |
| 9 | 173 | LBelt | SomMotB | 0.6214 | -0.0007 | -ADHD |
| 10 | 174 | A4 | SomMotB | 0.6152 | -0.0004 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0010 |
| 111 | AAIC | SalVentAttnB | +0.0007 |
| 114 | FOP2 | SomMotB | +0.0007 |
| 112 | FOP1 | SalVentAttnA | +0.0006 |
| 166 | PoI1 | SalVentAttnA | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0013 |
| 22 | MT | DorsAttnA | -0.0010 |
| 136 | PHT | ContB | -0.0010 |
| 152 | VMV1 | VisPeri | -0.0009 |
| 117 | EC | Limbic_TempPole | -0.0009 |

### Cluster 2 — Top 10 ROIs by Importance (N=110)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5397 | -0.0000 | -ADHD |
| 2 | 134 | TF | Limbic_TempPole | 0.5266 | -0.0002 | -ADHD |
| 3 | 4 | V3 | VisCent | 0.5200 | +0.0001 | +ADHD |
| 4 | 159 | VMV2 | VisPeri | 0.5174 | +0.0003 | +ADHD |
| 5 | 175 | STSva | TempPar | 0.5124 | -0.0002 | -ADHD |
| 6 | 149 | PGi | DefaultC | 0.5102 | +0.0001 | +ADHD |
| 7 | 12 | V3A | VisPeri | 0.5094 | -0.0001 | -ADHD |
| 8 | 129 | STSvp | TempPar | 0.4980 | +0.0000 | +ADHD |
| 9 | 34 | 31pv | DefaultC | 0.4955 | -0.0003 | -ADHD |
| 10 | 126 | PHA3 | DefaultB | 0.4942 | -0.0006 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0010 |
| 120 | ProS | VisPeri | +0.0006 |
| 177 | PI | SalVentAttnA | +0.0005 |
| 91 | 13l | Limbic_OFC | +0.0004 |
| 176 | TE1m | TempPar | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0007 |
| 151 | V6A | DorsAttnA | -0.0006 |
| 126 | PHA3 | DefaultB | -0.0006 |
| 94 | LIPd | ContB | -0.0006 |
| 21 | PIT | VisCent | -0.0005 |

### Cluster 3 — Top 10 ROIs by Importance (N=42)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 70 | 9p | TempPar | 0.8660 | -0.0008 | -ADHD |
| 2 | 23 | A1 | SomMotB | 0.8369 | +0.0008 | +ADHD |
| 3 | 39 | 24dd | SomMotA | 0.8074 | -0.0015 | -ADHD |
| 4 | 80 | IFSp | ContB | 0.8060 | -0.0003 | -ADHD |
| 5 | 97 | s6-8 | DefaultC | 0.7771 | -0.0010 | -ADHD |
| 6 | 78 | IFJa | ContB | 0.7674 | +0.0002 | +ADHD |
| 7 | 43 | 6ma | SalVentAttnA | 0.7623 | -0.0016 | -ADHD |
| 8 | 179 | p24 | DefaultC | 0.7603 | +0.0008 | +ADHD |
| 9 | 54 | 6mp | SomMotA | 0.7596 | -0.0014 | -ADHD |
| 10 | 178 | a32pr | SalVentAttnB | 0.7510 | -0.0009 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0033 |
| 154 | PHA2 | DefaultB | +0.0015 |
| 165 | pOFC | Limbic_OFC | +0.0014 |
| 164 | s32 | Limbic_OFC | +0.0013 |
| 163 | 25 | Limbic_OFC | +0.0013 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 34 | 31pv | DefaultC | -0.0023 |
| 41 | 7AL | DorsAttnB | -0.0022 |
| 38 | 5L | DorsAttnB | -0.0019 |
| 35 | 5m | SomMotA | -0.0019 |
| 161 | 31a | ContA | -0.0019 |

### Cluster 4 — Top 10 ROIs by Importance (N=5)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 163 | 25 | Limbic_OFC | 2.0743 | +0.0108 | +ADHD |
| 2 | 177 | PI | SalVentAttnA | 1.9758 | +0.0337 | +ADHD |
| 3 | 165 | pOFC | Limbic_OFC | 1.9588 | +0.0212 | +ADHD |
| 4 | 92 | OFC | Limbic_OFC | 1.7879 | +0.0055 | +ADHD |
| 5 | 171 | TGv | Limbic_TempPole | 1.7538 | +0.1289 | +ADHD |
| 6 | 164 | s32 | Limbic_OFC | 1.6740 | +0.0039 | +ADHD |
| 7 | 60 | a24 | DefaultC | 1.6711 | +0.0052 | +ADHD |
| 8 | 64 | 10r | DefaultC | 1.5996 | +0.0046 | +ADHD |
| 9 | 117 | EC | Limbic_TempPole | 1.5594 | +0.0539 | +ADHD |
| 10 | 106 | TA2 | SomMotB | 1.5395 | +0.0237 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.1289 |
| 121 | PeEc | Limbic_TempPole | +0.1102 |
| 134 | TF | Limbic_TempPole | +0.0890 |
| 130 | TGd | Limbic_TempPole | +0.0738 |
| 117 | EC | Limbic_TempPole | +0.0539 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 69 | 8BL | TempPar | -0.0058 |
| 102 | 52 | SomMotB | -0.0056 |
| 101 | OP2-3 | SomMotB | -0.0050 |
| 114 | FOP2 | SomMotB | -0.0049 |
| 40 | 24dv | SalVentAttnA | -0.0043 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.8118 | 0.4901 | 0.3885 | 0.6082 | 1.2700 |
| ContB | 0.7973 | 0.5384 | 0.4289 | 0.6709 | 1.0416 |
| ContC | 0.7560 | 0.5263 | 0.4272 | 0.6398 | 1.0844 |
| DefaultA | 0.7193 | 0.5726 | 0.4286 | 0.6471 | 1.0690 |
| DefaultB | 0.7356 | 0.5522 | 0.4450 | 0.6578 | 0.9762 |
| DefaultC | 0.8949 | 0.5477 | 0.4392 | 0.6799 | 1.2961 |
| DorsAttnA | 0.7294 | 0.5138 | 0.4291 | 0.5744 | 1.1010 |
| DorsAttnB | 0.8372 | 0.5020 | 0.4016 | 0.6564 | 1.1425 |
| Limbic_OFC | 0.9877 | 0.4820 | 0.3793 | 0.6310 | 1.6405 |
| Limbic_TempPole | 0.7685 | 0.5129 | 0.3998 | 0.6233 | 1.3460 |
| SalVentAttnA | 0.8362 | 0.5093 | 0.3952 | 0.6564 | 1.1313 |
| SalVentAttnB | 0.8393 | 0.5314 | 0.4169 | 0.6818 | 1.1215 |
| SomMotA | 0.9089 | 0.5032 | 0.3766 | 0.6973 | 1.2907 |
| SomMotB | 0.7972 | 0.5486 | 0.4124 | 0.6704 | 1.0650 |
| TempPar | 0.7966 | 0.5410 | 0.4327 | 0.6556 | 1.0833 |
| VisCent | 0.6303 | 0.5287 | 0.4616 | 0.5544 | 0.8615 |
| VisPeri | 0.6577 | 0.5158 | 0.4500 | 0.5713 | 0.8978 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | -0.0007 (-ADHD) | -0.0001 (-ADHD) | -0.0003 (-ADHD) | -0.0009 (-ADHD) | +0.0010 (+ADHD) |
| ContB | -0.0011 (-ADHD) | -0.0001 (-ADHD) | -0.0003 (-ADHD) | -0.0007 (-ADHD) | -0.0010 (-ADHD) |
| ContC | -0.0001 (-ADHD) | -0.0002 (-ADHD) | +0.0000 (+ADHD) | -0.0011 (-ADHD) | +0.0023 (+ADHD) |
| DefaultA | -0.0009 (-ADHD) | -0.0002 (-ADHD) | -0.0004 (-ADHD) | -0.0008 (-ADHD) | +0.0092 (+ADHD) |
| DefaultB | -0.0007 (-ADHD) | -0.0004 (-ADHD) | -0.0000 (-ADHD) | +0.0005 (+ADHD) | +0.0042 (+ADHD) |
| DefaultC | -0.0007 (-ADHD) | -0.0001 (-ADHD) | -0.0000 (-ADHD) | -0.0007 (-ADHD) | +0.0009 (+ADHD) |
| DorsAttnA | -0.0008 (-ADHD) | -0.0005 (-ADHD) | -0.0002 (-ADHD) | -0.0001 (-ADHD) | +0.0051 (+ADHD) |
| DorsAttnB | -0.0008 (-ADHD) | +0.0000 (+ADHD) | -0.0002 (-ADHD) | -0.0014 (-ADHD) | -0.0010 (-ADHD) |
| Limbic_OFC | +0.0016 (+ADHD) | +0.0001 (+ADHD) | +0.0002 (+ADHD) | +0.0015 (+ADHD) | +0.0100 (+ADHD) |
| Limbic_TempPole | -0.0001 (-ADHD) | -0.0003 (-ADHD) | +0.0002 (+ADHD) | +0.0004 (+ADHD) | +0.0649 (+ADHD) |
| SalVentAttnA | -0.0007 (-ADHD) | +0.0003 (+ADHD) | +0.0000 (+ADHD) | -0.0006 (-ADHD) | +0.0004 (+ADHD) |
| SalVentAttnB | -0.0007 (-ADHD) | +0.0004 (+ADHD) | -0.0000 (-ADHD) | -0.0005 (-ADHD) | +0.0023 (+ADHD) |
| SomMotA | -0.0007 (-ADHD) | -0.0003 (-ADHD) | -0.0002 (-ADHD) | -0.0016 (-ADHD) | -0.0006 (-ADHD) |
| SomMotB | -0.0012 (-ADHD) | -0.0001 (-ADHD) | -0.0001 (-ADHD) | -0.0004 (-ADHD) | -0.0002 (-ADHD) |
| TempPar | -0.0004 (-ADHD) | -0.0000 (-ADHD) | +0.0000 (+ADHD) | -0.0004 (-ADHD) | +0.0056 (+ADHD) |
| VisCent | +0.0001 (+ADHD) | -0.0005 (-ADHD) | -0.0001 (-ADHD) | -0.0004 (-ADHD) | +0.0036 (+ADHD) |
| VisPeri | -0.0002 (-ADHD) | -0.0003 (-ADHD) | +0.0002 (+ADHD) | +0.0004 (+ADHD) | +0.0036 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
