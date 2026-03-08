# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_3 | **K**: 3 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.3881

### Cluster Profiles

| Cluster | N | Dominant Circuit | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:----------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 139 | Executive | 0.249±0.001 | 0.252±0.000 | 0.248±0.001 | 0.252±0.001 |
| 1 | 6 | DMN | 0.255±0.002 | 0.252±0.001 | 0.250±0.001 | 0.243±0.003 |
| 2 | 143 | Executive | 0.250±0.001 | 0.252±0.000 | 0.248±0.001 | 0.250±0.001 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| DMN | 155.754 | 0.0000 | Yes |
| Executive | 6.871 | 0.0012 | Yes |
| Salience | 45.914 | 0.0000 | Yes |
| SensoriMotor | 388.619 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3955

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | DMN | Executive | Salience | SensoriMotor |
|:-------:|:-:|:---------------:|:-----:|:-----:|:-----:|:-----:|
| 0 | 78 | DMN | 3.750 | 3.340 | 3.399 | 2.324 |
| 1 | 12 | DMN | 3.410 | 3.408 | 3.153 | 2.506 |
| 2 | 198 | DMN | 3.879 | 3.393 | 3.639 | 2.445 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.2411

### Cluster 0 — Top 10 ROIs by Importance (N=94)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 52 | 3a | SomMotA | 0.7116 | -0.0014 | -ADHD |
| 2 | 35 | 5m | SomMotA | 0.7027 | -0.0019 | -ADHD |
| 3 | 70 | 9p | TempPar | 0.6906 | -0.0008 | -ADHD |
| 4 | 101 | OP2-3 | SomMotB | 0.6816 | -0.0002 | -ADHD |
| 5 | 32 | v23ab | DefaultC | 0.6813 | -0.0013 | -ADHD |
| 6 | 103 | RI | SomMotB | 0.6763 | -0.0007 | -ADHD |
| 7 | 166 | PoI1 | SalVentAttnA | 0.6750 | -0.0012 | -ADHD |
| 8 | 97 | s6-8 | DefaultC | 0.6742 | -0.0011 | -ADHD |
| 9 | 67 | 8Ad | DefaultC | 0.6720 | -0.0011 | -ADHD |
| 10 | 8 | 3b | SomMotA | 0.6709 | -0.0016 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0013 |
| 91 | 13l | Limbic_OFC | +0.0007 |
| 89 | 10pp | Limbic_OFC | +0.0006 |
| 87 | 10v | Limbic_OFC | +0.0006 |
| 165 | pOFC | Limbic_OFC | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 35 | 5m | SomMotA | -0.0019 |
| 129 | STSvp | TempPar | -0.0017 |
| 160 | 31pd | DefaultC | -0.0017 |
| 34 | 31pv | DefaultC | -0.0017 |
| 8 | 3b | SomMotA | -0.0016 |

### Cluster 1 — Top 10 ROIs by Importance (N=170)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 32 | v23ab | DefaultC | 0.5673 | +0.0002 | +ADHD |
| 2 | 15 | V7 | VisCent | 0.5485 | -0.0002 | -ADHD |
| 3 | 157 | V3CD | VisCent | 0.5474 | -0.0003 | -ADHD |
| 4 | 20 | LO2 | VisCent | 0.5430 | -0.0000 | -ADHD |
| 5 | 176 | TE1m | TempPar | 0.5328 | +0.0003 | +ADHD |
| 6 | 175 | STSva | TempPar | 0.5240 | -0.0001 | -ADHD |
| 7 | 65 | 47m | TempPar | 0.5198 | +0.0001 | +ADHD |
| 8 | 3 | V2 | VisPeri | 0.5167 | -0.0000 | -ADHD |
| 9 | 19 | LO1 | VisCent | 0.5158 | +0.0002 | +ADHD |
| 10 | 126 | PHA3 | DefaultB | 0.5124 | -0.0004 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 177 | PI | SalVentAttnA | +0.0008 |
| 119 | H | Limbic_TempPole | +0.0008 |
| 120 | ProS | VisPeri | +0.0006 |
| 167 | Ig | SomMotB | +0.0006 |
| 100 | OP1 | SomMotB | +0.0005 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 144 | IP1 | ContB | -0.0006 |
| 136 | PHT | ContB | -0.0006 |
| 143 | IP2 | ContB | -0.0005 |
| 122 | STGa | DefaultA | -0.0005 |
| 28 | 7Pm | ContA | -0.0005 |

### Cluster 2 — Top 10 ROIs by Importance (N=24)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 163 | 25 | Limbic_OFC | 1.2592 | +0.0051 | +ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.2142 | +0.0029 | +ADHD |
| 3 | 91 | 13l | Limbic_OFC | 1.1939 | +0.0099 | +ADHD |
| 4 | 164 | s32 | Limbic_OFC | 1.1795 | +0.0030 | +ADHD |
| 5 | 35 | 5m | SomMotA | 1.1781 | +0.0007 | +ADHD |
| 6 | 165 | pOFC | Limbic_OFC | 1.1675 | +0.0072 | +ADHD |
| 7 | 60 | a24 | DefaultC | 1.1483 | +0.0044 | +ADHD |
| 8 | 52 | 3a | SomMotA | 1.0923 | +0.0003 | +ADHD |
| 9 | 37 | 23c | SalVentAttnA | 1.0896 | +0.0011 | +ADHD |
| 10 | 31 | 23d | DefaultC | 1.0849 | +0.0012 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 121 | PeEc | Limbic_TempPole | +0.0257 |
| 171 | TGv | Limbic_TempPole | +0.0232 |
| 130 | TGd | Limbic_TempPole | +0.0183 |
| 134 | TF | Limbic_TempPole | +0.0162 |
| 117 | EC | Limbic_TempPole | +0.0112 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0019 |
| 123 | PBelt | SomMotB | -0.0010 |
| 172 | MBelt | SomMotB | -0.0009 |
| 69 | 8BL | TempPar | -0.0008 |
| 84 | a9-46v | ContC | -0.0007 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | 0.5598 | 0.4043 | 0.9018 |
| ContB | 0.5667 | 0.4154 | 0.7924 |
| ContC | 0.5219 | 0.3900 | 0.7224 |
| DefaultA | 0.6318 | 0.4710 | 0.8182 |
| DefaultB | 0.6037 | 0.4623 | 0.8014 |
| DefaultC | 0.6243 | 0.4518 | 0.9729 |
| DorsAttnA | 0.5431 | 0.4289 | 0.7871 |
| DorsAttnB | 0.5666 | 0.3996 | 0.8599 |
| Limbic_OFC | 0.5909 | 0.3980 | 1.1417 |
| Limbic_TempPole | 0.5752 | 0.4180 | 0.8946 |
| SalVentAttnA | 0.5980 | 0.4206 | 0.9022 |
| SalVentAttnB | 0.6122 | 0.4347 | 0.8873 |
| SomMotA | 0.6528 | 0.4209 | 1.0389 |
| SomMotB | 0.6090 | 0.4393 | 0.8561 |
| TempPar | 0.6160 | 0.4586 | 0.8647 |
| VisCent | 0.5679 | 0.4972 | 0.7212 |
| VisPeri | 0.5548 | 0.4681 | 0.7334 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:--------:|:--------:|:--------:|
| ContA | -0.0010 (-ADHD) | -0.0001 (-ADHD) | +0.0008 (+ADHD) |
| ContB | -0.0008 (-ADHD) | -0.0001 (-ADHD) | +0.0001 (+ADHD) |
| ContC | -0.0009 (-ADHD) | +0.0000 (+ADHD) | +0.0010 (+ADHD) |
| DefaultA | -0.0010 (-ADHD) | -0.0002 (-ADHD) | +0.0022 (+ADHD) |
| DefaultB | -0.0007 (-ADHD) | -0.0000 (-ADHD) | +0.0016 (+ADHD) |
| DefaultC | -0.0009 (-ADHD) | +0.0000 (+ADHD) | +0.0009 (+ADHD) |
| DorsAttnA | -0.0006 (-ADHD) | -0.0002 (-ADHD) | +0.0013 (+ADHD) |
| DorsAttnB | -0.0011 (-ADHD) | -0.0000 (-ADHD) | +0.0000 (+ADHD) |
| Limbic_OFC | +0.0006 (+ADHD) | +0.0001 (+ADHD) | +0.0049 (+ADHD) |
| Limbic_TempPole | -0.0003 (-ADHD) | +0.0000 (+ADHD) | +0.0140 (+ADHD) |
| SalVentAttnA | -0.0006 (-ADHD) | +0.0002 (+ADHD) | +0.0006 (+ADHD) |
| SalVentAttnB | -0.0006 (-ADHD) | +0.0002 (+ADHD) | +0.0011 (+ADHD) |
| SomMotA | -0.0013 (-ADHD) | -0.0002 (-ADHD) | +0.0004 (+ADHD) |
| SomMotB | -0.0009 (-ADHD) | +0.0001 (+ADHD) | +0.0001 (+ADHD) |
| TempPar | -0.0007 (-ADHD) | +0.0001 (+ADHD) | +0.0018 (+ADHD) |
| VisCent | -0.0008 (-ADHD) | -0.0001 (-ADHD) | +0.0015 (+ADHD) |
| VisPeri | -0.0003 (-ADHD) | +0.0001 (+ADHD) | +0.0013 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
