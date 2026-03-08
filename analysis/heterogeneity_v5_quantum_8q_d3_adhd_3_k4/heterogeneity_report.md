# ADHD Heterogeneity Analysis via Learned Representations

**Model**: quantum | **Config**: adhd_3 | **K**: 4 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.4452

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 135 | SensoriMotor | 0.248±0.001 | 0.250±0.001 | 0.249±0.001 | 0.252±0.000 |
| 1 | 7 | Salience | 0.241±0.002 | 0.239±0.003 | 0.267±0.004 | 0.254±0.002 |
| 2 | 1 | Executive | 0.262±0.000 | 0.268±0.000 | 0.224±0.000 | 0.246±0.000 |
| 3 | 145 | Executive | 0.249±0.001 | 0.252±0.001 | 0.246±0.001 | 0.252±0.001 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 286.542 | 0.0000 | Yes |
| Executive | 345.781 | 0.0000 | Yes |
| Salience | 575.050 | 0.0000 | Yes |
| SensoriMotor | 60.331 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.1595

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 71 | DMN | 1.645 | 1.265 | 1.611 | 1.216 |
| 1 | 97 | Salience | 2.098 | 1.732 | 2.587 | 1.265 |
| 2 | 65 | Executive | 1.163 | 1.357 | 1.144 | 1.282 |
| 3 | 55 | Executive | 1.395 | 1.899 | 1.490 | 1.176 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1603

### Cluster 0 — Top 10 ROIs by Importance (N=109)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 70 | 9p | TempPar | 0.6695 | -0.0001 | -ADHD |
| 2 | 80 | IFSp | ContB | 0.6635 | -0.0004 | -ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.6596 | -0.0010 | -ADHD |
| 4 | 128 | STSdp | DefaultA | 0.6540 | -0.0005 | -ADHD |
| 5 | 23 | A1 | SomMotB | 0.6518 | -0.0003 | -ADHD |
| 6 | 32 | v23ab | DefaultC | 0.6509 | +0.0000 | +ADHD |
| 7 | 149 | PGi | DefaultC | 0.6467 | -0.0002 | -ADHD |
| 8 | 175 | STSva | TempPar | 0.6434 | -0.0007 | -ADHD |
| 9 | 77 | 6r | ContB | 0.6389 | +0.0001 | +ADHD |
| 10 | 173 | LBelt | SomMotB | 0.6371 | -0.0007 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0007 |
| 93 | 47s | TempPar | +0.0007 |
| 171 | TGv | Limbic_TempPole | +0.0007 |
| 131 | TE1a | TempPar | +0.0006 |
| 112 | FOP1 | SalVentAttnA | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0012 |
| 117 | EC | Limbic_TempPole | -0.0010 |
| 136 | PHT | ContB | -0.0010 |
| 134 | TF | Limbic_TempPole | -0.0010 |
| 142 | PGp | DorsAttnA | -0.0010 |

### Cluster 1 — Top 10 ROIs by Importance (N=8)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.8194 | +0.0037 | +ADHD |
| 2 | 163 | 25 | Limbic_OFC | 1.7391 | +0.0068 | +ADHD |
| 3 | 165 | pOFC | Limbic_OFC | 1.7266 | +0.0129 | +ADHD |
| 4 | 177 | PI | SalVentAttnA | 1.7059 | +0.0219 | +ADHD |
| 5 | 164 | s32 | Limbic_OFC | 1.6621 | +0.0023 | +ADHD |
| 6 | 171 | TGv | Limbic_TempPole | 1.5145 | +0.0812 | +ADHD |
| 7 | 39 | 24dd | SomMotA | 1.4908 | +0.0008 | +ADHD |
| 8 | 64 | 10r | DefaultC | 1.4808 | +0.0030 | +ADHD |
| 9 | 60 | a24 | DefaultC | 1.4726 | +0.0050 | +ADHD |
| 10 | 117 | EC | Limbic_TempPole | 1.4103 | +0.0336 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.0812 |
| 121 | PeEc | Limbic_TempPole | +0.0698 |
| 134 | TF | Limbic_TempPole | +0.0559 |
| 130 | TGd | Limbic_TempPole | +0.0467 |
| 117 | EC | Limbic_TempPole | +0.0336 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0032 |
| 69 | 8BL | TempPar | -0.0031 |
| 114 | FOP2 | SomMotB | -0.0022 |
| 101 | OP2-3 | SomMotB | -0.0022 |
| 123 | PBelt | SomMotB | -0.0022 |

### Cluster 2 — Top 10 ROIs by Importance (N=49)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.0244 | +0.0031 | +ADHD |
| 2 | 70 | 9p | TempPar | 0.9853 | -0.0012 | -ADHD |
| 3 | 39 | 24dd | SomMotA | 0.9372 | -0.0013 | -ADHD |
| 4 | 165 | pOFC | Limbic_OFC | 0.9118 | +0.0024 | +ADHD |
| 5 | 179 | p24 | DefaultC | 0.8891 | +0.0003 | +ADHD |
| 6 | 163 | 25 | Limbic_OFC | 0.8866 | +0.0018 | +ADHD |
| 7 | 54 | 6mp | SomMotA | 0.8832 | -0.0017 | -ADHD |
| 8 | 43 | 6ma | SalVentAttnA | 0.8754 | -0.0019 | -ADHD |
| 9 | 40 | 24dv | SalVentAttnA | 0.8721 | -0.0008 | -ADHD |
| 10 | 23 | A1 | SomMotB | 0.8720 | -0.0008 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0031 |
| 91 | 13l | Limbic_OFC | +0.0027 |
| 165 | pOFC | Limbic_OFC | +0.0024 |
| 163 | 25 | Limbic_OFC | +0.0018 |
| 88 | a10p | ContC | +0.0018 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 38 | 5L | DorsAttnB | -0.0020 |
| 143 | IP2 | ContB | -0.0020 |
| 43 | 6ma | SalVentAttnA | -0.0019 |
| 35 | 5m | SomMotA | -0.0019 |
| 173 | LBelt | SomMotB | -0.0019 |

### Cluster 3 — Top 10 ROIs by Importance (N=122)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5420 | -0.0001 | -ADHD |
| 2 | 134 | TF | Limbic_TempPole | 0.5401 | -0.0002 | -ADHD |
| 3 | 4 | V3 | VisCent | 0.5265 | -0.0001 | -ADHD |
| 4 | 159 | VMV2 | VisPeri | 0.5230 | +0.0002 | +ADHD |
| 5 | 149 | PGi | DefaultC | 0.5206 | +0.0000 | +ADHD |
| 6 | 175 | STSva | TempPar | 0.5176 | -0.0001 | -ADHD |
| 7 | 12 | V3A | VisPeri | 0.5150 | -0.0002 | -ADHD |
| 8 | 126 | PHA3 | DefaultB | 0.5063 | -0.0005 | -ADHD |
| 9 | 129 | STSvp | TempPar | 0.5045 | -0.0001 | -ADHD |
| 10 | 128 | STSdp | DefaultA | 0.5027 | -0.0002 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0008 |
| 120 | ProS | VisPeri | +0.0006 |
| 91 | 13l | Limbic_OFC | +0.0005 |
| 60 | a24 | DefaultC | +0.0004 |
| 177 | PI | SalVentAttnA | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0008 |
| 21 | PIT | VisCent | -0.0006 |
| 50 | 1 | SomMotA | -0.0006 |
| 35 | 5m | SomMotA | -0.0005 |
| 126 | PHA3 | DefaultB | -0.0005 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.5075 | 1.1783 | 0.7042 | 0.3931 |
| ContB | 0.5612 | 0.9646 | 0.7367 | 0.4332 |
| ContC | 0.5458 | 1.0095 | 0.6939 | 0.4314 |
| DefaultA | 0.5846 | 0.9487 | 0.6875 | 0.4385 |
| DefaultB | 0.5670 | 0.9371 | 0.7005 | 0.4503 |
| DefaultC | 0.5670 | 1.1901 | 0.7856 | 0.4451 |
| DorsAttnA | 0.5221 | 0.9839 | 0.6530 | 0.4342 |
| DorsAttnB | 0.5265 | 1.0695 | 0.7448 | 0.4057 |
| Limbic_OFC | 0.4934 | 1.5448 | 0.8002 | 0.3882 |
| Limbic_TempPole | 0.5297 | 1.2370 | 0.6812 | 0.4074 |
| SalVentAttnA | 0.5322 | 1.0704 | 0.7440 | 0.4007 |
| SalVentAttnB | 0.5549 | 1.0478 | 0.7637 | 0.4216 |
| SomMotA | 0.5348 | 1.2227 | 0.7976 | 0.3810 |
| SomMotB | 0.5667 | 1.0030 | 0.7351 | 0.4210 |
| TempPar | 0.5596 | 0.9968 | 0.7283 | 0.4378 |
| VisCent | 0.5295 | 0.7932 | 0.5927 | 0.4685 |
| VisPeri | 0.5237 | 0.8270 | 0.6138 | 0.4546 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | -0.0001 (-ADHD) | +0.0015 (+ADHD) | -0.0011 (-ADHD) | -0.0003 (-ADHD) |
| ContB | -0.0003 (-ADHD) | -0.0000 (-ADHD) | -0.0009 (-ADHD) | -0.0002 (-ADHD) |
| ContC | -0.0004 (-ADHD) | +0.0024 (+ADHD) | -0.0007 (-ADHD) | +0.0000 (+ADHD) |
| DefaultA | -0.0003 (-ADHD) | +0.0059 (+ADHD) | -0.0008 (-ADHD) | -0.0004 (-ADHD) |
| DefaultB | -0.0003 (-ADHD) | +0.0033 (+ADHD) | -0.0002 (-ADHD) | -0.0000 (-ADHD) |
| DefaultC | -0.0002 (-ADHD) | +0.0013 (+ADHD) | -0.0008 (-ADHD) | -0.0000 (-ADHD) |
| DorsAttnA | -0.0004 (-ADHD) | +0.0036 (+ADHD) | -0.0006 (-ADHD) | -0.0002 (-ADHD) |
| DorsAttnB | -0.0002 (-ADHD) | -0.0002 (-ADHD) | -0.0014 (-ADHD) | -0.0002 (-ADHD) |
| Limbic_OFC | +0.0001 (+ADHD) | +0.0068 (+ADHD) | +0.0020 (+ADHD) | +0.0002 (+ADHD) |
| Limbic_TempPole | -0.0003 (-ADHD) | +0.0411 (+ADHD) | +0.0004 (+ADHD) | +0.0001 (+ADHD) |
| SalVentAttnA | +0.0002 (+ADHD) | +0.0008 (+ADHD) | -0.0008 (-ADHD) | +0.0000 (+ADHD) |
| SalVentAttnB | +0.0002 (+ADHD) | +0.0021 (+ADHD) | -0.0006 (-ADHD) | +0.0000 (+ADHD) |
| SomMotA | -0.0004 (-ADHD) | +0.0004 (+ADHD) | -0.0015 (-ADHD) | -0.0003 (-ADHD) |
| SomMotB | -0.0001 (-ADHD) | +0.0004 (+ADHD) | -0.0010 (-ADHD) | -0.0001 (-ADHD) |
| TempPar | -0.0001 (-ADHD) | +0.0041 (+ADHD) | -0.0005 (-ADHD) | +0.0000 (+ADHD) |
| VisCent | -0.0005 (-ADHD) | +0.0027 (+ADHD) | -0.0001 (-ADHD) | -0.0002 (-ADHD) |
| VisPeri | -0.0002 (-ADHD) | +0.0026 (+ADHD) | +0.0002 (+ADHD) | +0.0000 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
