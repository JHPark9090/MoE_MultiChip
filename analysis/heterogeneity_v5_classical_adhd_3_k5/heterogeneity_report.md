# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_3 | **K**: 5 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.3970

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 134 | Executive | 0.249±0.000 | 0.252±0.000 | 0.248±0.000 | 0.251±0.000 |
| 1 | 6 | DMN | 0.255±0.002 | 0.252±0.001 | 0.250±0.001 | 0.243±0.003 |
| 2 | 1 | DMN | 0.259±0.000 | 0.248±0.000 | 0.241±0.000 | 0.252±0.000 |
| 3 | 134 | Executive | 0.250±0.001 | 0.252±0.000 | 0.248±0.000 | 0.250±0.001 |
| 4 | 13 | SensoriMotor | 0.248±0.002 | 0.252±0.001 | 0.247±0.001 | 0.253±0.001 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 191.086 | 0.0000 | Yes |
| Executive | 43.858 | 0.0000 | Yes |
| Salience | 81.053 | 0.0000 | Yes |
| SensoriMotor | 320.272 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3100

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 162 | DMN | 3.882 | 3.394 | 3.655 | 2.457 |
| 1 | 10 | DMN | 3.405 | 3.387 | 3.135 | 2.449 |
| 2 | 19 | DMN | 3.554 | 3.253 | 3.209 | 2.271 |
| 3 | 96 | DMN | 3.826 | 3.374 | 3.495 | 2.360 |
| 4 | 1 | Executive | 3.740 | 3.792 | 3.374 | 3.267 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1338

### Cluster 0 — Top 10 ROIs by Importance (N=19)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 92 | OFC | Limbic_OFC | 1.1307 | +0.0025 | +ADHD |
| 2 | 35 | 5m | SomMotA | 1.0918 | +0.0009 | +ADHD |
| 3 | 164 | s32 | Limbic_OFC | 1.0892 | +0.0029 | +ADHD |
| 4 | 91 | 13l | Limbic_OFC | 1.0815 | +0.0063 | +ADHD |
| 5 | 163 | 25 | Limbic_OFC | 1.0735 | +0.0038 | +ADHD |
| 6 | 52 | 3a | SomMotA | 1.0404 | +0.0005 | +ADHD |
| 7 | 31 | 23d | DefaultC | 1.0295 | +0.0013 | +ADHD |
| 8 | 60 | a24 | DefaultC | 1.0135 | +0.0043 | +ADHD |
| 9 | 165 | pOFC | Limbic_OFC | 1.0113 | +0.0041 | +ADHD |
| 10 | 37 | 23c | SalVentAttnA | 1.0086 | +0.0014 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 91 | 13l | Limbic_OFC | +0.0063 |
| 88 | a10p | ContC | +0.0052 |
| 60 | a24 | DefaultC | +0.0043 |
| 165 | pOFC | Limbic_OFC | +0.0041 |
| 163 | 25 | Limbic_OFC | +0.0038 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0009 |
| 84 | a9-46v | ContC | -0.0007 |
| 123 | PBelt | SomMotB | -0.0005 |
| 172 | MBelt | SomMotB | -0.0004 |
| 55 | 6v | DorsAttnB | -0.0004 |

### Cluster 1 — Top 10 ROIs by Importance (N=108)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.6370 | +0.0003 | +ADHD |
| 2 | 124 | A5 | DefaultA | 0.6045 | +0.0000 | +ADHD |
| 3 | 175 | STSva | TempPar | 0.5907 | -0.0005 | -ADHD |
| 4 | 15 | V7 | VisCent | 0.5894 | -0.0004 | -ADHD |
| 5 | 20 | LO2 | VisCent | 0.5887 | -0.0005 | -ADHD |
| 6 | 103 | RI | SomMotB | 0.5848 | -0.0003 | -ADHD |
| 7 | 126 | PHA3 | DefaultB | 0.5824 | -0.0006 | -ADHD |
| 8 | 65 | 47m | TempPar | 0.5821 | -0.0005 | -ADHD |
| 9 | 173 | LBelt | SomMotB | 0.5792 | -0.0006 | -ADHD |
| 10 | 73 | 44 | TempPar | 0.5784 | +0.0002 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0009 |
| 166 | PoI1 | SalVentAttnA | +0.0007 |
| 114 | FOP2 | SomMotB | +0.0007 |
| 111 | AAIC | SalVentAttnB | +0.0006 |
| 112 | FOP1 | SalVentAttnA | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 162 | VVC | VisCent | -0.0013 |
| 22 | MT | DorsAttnA | -0.0010 |
| 136 | PHT | ContB | -0.0010 |
| 152 | VMV1 | VisPeri | -0.0010 |
| 121 | PeEc | Limbic_TempPole | -0.0009 |

### Cluster 2 — Top 10 ROIs by Importance (N=108)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5408 | -0.0000 | -ADHD |
| 2 | 15 | V7 | VisCent | 0.5406 | -0.0001 | -ADHD |
| 3 | 157 | V3CD | VisCent | 0.5393 | -0.0003 | -ADHD |
| 4 | 20 | LO2 | VisCent | 0.5134 | -0.0001 | -ADHD |
| 5 | 176 | TE1m | TempPar | 0.5123 | +0.0005 | +ADHD |
| 6 | 65 | 47m | TempPar | 0.4960 | +0.0002 | +ADHD |
| 7 | 3 | V2 | VisPeri | 0.4911 | +0.0001 | +ADHD |
| 8 | 175 | STSva | TempPar | 0.4881 | -0.0001 | -ADHD |
| 9 | 19 | LO1 | VisCent | 0.4870 | +0.0002 | +ADHD |
| 10 | 12 | V3A | VisPeri | 0.4809 | -0.0001 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0011 |
| 120 | ProS | VisPeri | +0.0007 |
| 91 | 13l | Limbic_OFC | +0.0005 |
| 121 | PeEc | Limbic_TempPole | +0.0005 |
| 176 | TE1m | TempPar | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0006 |
| 151 | V6A | DorsAttnA | -0.0006 |
| 35 | 5m | SomMotA | -0.0006 |
| 126 | PHA3 | DefaultB | -0.0005 |
| 21 | PIT | VisCent | -0.0005 |

### Cluster 3 — Top 10 ROIs by Importance (N=48)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 35 | 5m | SomMotA | 0.7948 | -0.0029 | -ADHD |
| 2 | 52 | 3a | SomMotA | 0.7941 | -0.0019 | -ADHD |
| 3 | 70 | 9p | TempPar | 0.7758 | -0.0013 | -ADHD |
| 4 | 166 | PoI1 | SalVentAttnA | 0.7623 | -0.0023 | -ADHD |
| 5 | 101 | OP2-3 | SomMotB | 0.7594 | -0.0004 | -ADHD |
| 6 | 67 | 8Ad | DefaultC | 0.7538 | -0.0013 | -ADHD |
| 7 | 37 | 23c | SalVentAttnA | 0.7485 | -0.0019 | -ADHD |
| 8 | 8 | 3b | SomMotA | 0.7437 | -0.0022 | -ADHD |
| 9 | 7 | 4 | SomMotA | 0.7392 | -0.0022 | -ADHD |
| 10 | 31 | 23d | DefaultC | 0.7390 | -0.0015 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0022 |
| 91 | 13l | Limbic_OFC | +0.0007 |
| 89 | 10pp | Limbic_OFC | +0.0007 |
| 159 | VMV2 | VisPeri | +0.0007 |
| 65 | 47m | TempPar | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 35 | 5m | SomMotA | -0.0029 |
| 34 | 31pv | DefaultC | -0.0025 |
| 51 | 2 | SomMotA | -0.0024 |
| 143 | IP2 | ContB | -0.0024 |
| 160 | 31pd | DefaultC | -0.0024 |

### Cluster 4 — Top 10 ROIs by Importance (N=5)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 163 | 25 | Limbic_OFC | 1.9648 | +0.0103 | +ADHD |
| 2 | 165 | pOFC | Limbic_OFC | 1.7611 | +0.0190 | +ADHD |
| 3 | 177 | PI | SalVentAttnA | 1.7571 | +0.0300 | +ADHD |
| 4 | 60 | a24 | DefaultC | 1.6606 | +0.0051 | +ADHD |
| 5 | 91 | 13l | Limbic_OFC | 1.6212 | +0.0235 | +ADHD |
| 6 | 130 | TGd | Limbic_TempPole | 1.5935 | +0.0778 | +ADHD |
| 7 | 121 | PeEc | Limbic_TempPole | 1.5418 | +0.1191 | +ADHD |
| 8 | 90 | 11l | Limbic_OFC | 1.5354 | +0.0116 | +ADHD |
| 9 | 92 | OFC | Limbic_OFC | 1.5314 | +0.0047 | +ADHD |
| 10 | 164 | s32 | Limbic_OFC | 1.5228 | +0.0035 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.1191 |
| 171 | TGv | Limbic_TempPole | +0.1067 |
| 134 | TF | Limbic_TempPole | +0.0783 |
| 130 | TGd | Limbic_TempPole | +0.0778 |
| 109 | Pir | Limbic_TempPole | +0.0477 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 101 | OP2-3 | SomMotB | -0.0061 |
| 69 | 8BL | TempPar | -0.0057 |
| 102 | 52 | SomMotB | -0.0054 |
| 114 | FOP2 | SomMotB | -0.0049 |
| 42 | SCEF | SalVentAttnA | -0.0038 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.8146 | 0.4761 | 0.3772 | 0.6080 | 1.2332 |
| ContB | 0.7493 | 0.4871 | 0.3879 | 0.6124 | 0.9562 |
| ContC | 0.6679 | 0.4532 | 0.3678 | 0.5562 | 0.9295 |
| DefaultA | 0.7508 | 0.5738 | 0.4276 | 0.6524 | 1.0742 |
| DefaultB | 0.7611 | 0.5351 | 0.4340 | 0.6393 | 0.9546 |
| DefaultC | 0.8984 | 0.5285 | 0.4244 | 0.6788 | 1.2557 |
| DorsAttnA | 0.7184 | 0.4889 | 0.4086 | 0.5635 | 1.0483 |
| DorsAttnB | 0.8067 | 0.4698 | 0.3746 | 0.6251 | 1.0623 |
| Limbic_OFC | 1.0187 | 0.4765 | 0.3749 | 0.6510 | 1.6088 |
| Limbic_TempPole | 0.7842 | 0.4971 | 0.3869 | 0.6179 | 1.3141 |
| SalVentAttnA | 0.8481 | 0.5011 | 0.3883 | 0.6598 | 1.1075 |
| SalVentAttnB | 0.8349 | 0.5133 | 0.4041 | 0.6744 | 1.0864 |
| SomMotA | 0.9645 | 0.5193 | 0.3866 | 0.7309 | 1.3216 |
| SomMotB | 0.8105 | 0.5315 | 0.3978 | 0.6574 | 1.0297 |
| TempPar | 0.8088 | 0.5343 | 0.4288 | 0.6638 | 1.0768 |
| VisCent | 0.6765 | 0.5427 | 0.4767 | 0.5796 | 0.8913 |
| VisPeri | 0.6891 | 0.5139 | 0.4486 | 0.5787 | 0.9017 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|:--------:|:--------:|:--------:|:--------:|:--------:|
| ContA | +0.0007 (+ADHD) | -0.0001 (-ADHD) | -0.0002 (-ADHD) | -0.0014 (-ADHD) | +0.0010 (+ADHD) |
| ContB | +0.0003 (+ADHD) | -0.0001 (-ADHD) | -0.0003 (-ADHD) | -0.0012 (-ADHD) | -0.0008 (-ADHD) |
| ContC | +0.0007 (+ADHD) | -0.0002 (-ADHD) | +0.0000 (+ADHD) | -0.0011 (-ADHD) | +0.0022 (+ADHD) |
| DefaultA | +0.0003 (+ADHD) | -0.0002 (-ADHD) | -0.0004 (-ADHD) | -0.0013 (-ADHD) | +0.0094 (+ADHD) |
| DefaultB | +0.0009 (+ADHD) | -0.0004 (-ADHD) | -0.0000 (-ADHD) | -0.0004 (-ADHD) | +0.0041 (+ADHD) |
| DefaultC | +0.0009 (+ADHD) | -0.0001 (-ADHD) | -0.0000 (-ADHD) | -0.0014 (-ADHD) | +0.0008 (+ADHD) |
| DorsAttnA | +0.0004 (+ADHD) | -0.0004 (-ADHD) | -0.0002 (-ADHD) | -0.0007 (-ADHD) | +0.0048 (+ADHD) |
| DorsAttnB | +0.0003 (+ADHD) | +0.0000 (+ADHD) | -0.0002 (-ADHD) | -0.0017 (-ADHD) | -0.0010 (-ADHD) |
| Limbic_OFC | +0.0036 (+ADHD) | +0.0001 (+ADHD) | +0.0003 (+ADHD) | +0.0007 (+ADHD) | +0.0098 (+ADHD) |
| Limbic_TempPole | +0.0012 (+ADHD) | -0.0004 (-ADHD) | +0.0002 (+ADHD) | -0.0001 (-ADHD) | +0.0626 (+ADHD) |
| SalVentAttnA | +0.0006 (+ADHD) | +0.0003 (+ADHD) | +0.0001 (+ADHD) | -0.0012 (-ADHD) | +0.0002 (+ADHD) |
| SalVentAttnB | +0.0008 (+ADHD) | +0.0004 (+ADHD) | -0.0000 (-ADHD) | -0.0011 (-ADHD) | +0.0023 (+ADHD) |
| SomMotA | +0.0007 (+ADHD) | -0.0003 (-ADHD) | -0.0002 (-ADHD) | -0.0021 (-ADHD) | -0.0006 (-ADHD) |
| SomMotB | +0.0003 (+ADHD) | -0.0001 (-ADHD) | -0.0000 (-ADHD) | -0.0011 (-ADHD) | -0.0004 (-ADHD) |
| TempPar | +0.0008 (+ADHD) | -0.0001 (-ADHD) | +0.0001 (+ADHD) | -0.0009 (-ADHD) | +0.0057 (+ADHD) |
| VisCent | +0.0009 (+ADHD) | -0.0005 (-ADHD) | -0.0001 (-ADHD) | -0.0007 (-ADHD) | +0.0037 (+ADHD) |
| VisPeri | +0.0007 (+ADHD) | -0.0003 (-ADHD) | +0.0002 (+ADHD) | -0.0000 (-ADHD) | +0.0038 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
