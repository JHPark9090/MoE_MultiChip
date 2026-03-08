# ADHD Heterogeneity Analysis via Learned Representations

**Model**: classical | **Config**: adhd_2 | **K**: 4 clusters

## Level 1: Circuit-Level Heterogeneity (Gate Weights)

Each ADHD+ subject has a K-dimensional gate weight vector — a 'circuit fingerprint' showing how the model routes their brain data through DMN, Executive, Salience, and SensoriMotor experts.

**Silhouette score**: 0.5557

### Cluster Profiles

| Cluster | N | Dominant Circuit | Internal | External |
|:-------:|:-:|:----------------:|:-----:|:-----:|
| 0 | 106 | Internal | 0.531±0.028 | 0.469±0.028 |
| 1 | 21 | External | 0.321±0.068 | 0.679±0.068 |
| 2 | 9 | Internal | 0.729±0.061 | 0.271±0.061 |
| 3 | 152 | External | 0.457±0.024 | 0.543±0.024 |

### Gate Weight ANOVA Across Clusters

| Circuit | F-stat | p-value | Significant? |
|---------|:------:|:-------:|:------------:|
| Internal | 451.320 | 0.0000 | Yes |
| External | 451.320 | 0.0000 | Yes |

## Level 2: Network-Level Heterogeneity (Expert Outputs)

Expert output vectors (K×H dimensions) capture task-optimized representations learned by each circuit expert. These are richer than gate weights — they encode *what* each expert learned, not just *how much* it was used.

**Silhouette score**: 0.3666

### Expert Activation Norms per Cluster

| Cluster | N | Dominant Expert | Internal | External |
|:-------:|:-:|:---------------:|:-----:|:-----:|
| 0 | 96 | Internal | 3.963 | 1.431 |
| 1 | 76 | Internal | 6.487 | 1.328 |
| 2 | 93 | Internal | 3.908 | 1.730 |
| 3 | 23 | Internal | 5.438 | 4.731 |

## Level 3: ROI-Level Heterogeneity (Input Projection Activations)

Per-ROI importance scores combine input magnitude with learned weight norms — revealing which of the 180 brain regions are most important for each ADHD subtype cluster.

**Silhouette score**: 0.1553

### Cluster 0 — Top 10 ROIs by Importance (N=50)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 38 | 5L | DorsAttnB | 1.1914 | -0.0029 | -ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.0939 | +0.0032 | +ADHD |
| 3 | 54 | 6mp | SomMotA | 1.0542 | -0.0020 | -ADHD |
| 4 | 161 | 31a | ContA | 1.0382 | -0.0021 | -ADHD |
| 5 | 171 | TGv | Limbic_TempPole | 1.0015 | +0.0005 | +ADHD |
| 6 | 168 | FOP5 | SalVentAttnB | 0.9862 | -0.0009 | -ADHD |
| 7 | 68 | 9m | TempPar | 0.9696 | -0.0000 | -ADHD |
| 8 | 73 | 44 | TempPar | 0.9668 | -0.0002 | -ADHD |
| 9 | 43 | 6ma | SalVentAttnA | 0.9592 | -0.0020 | -ADHD |
| 10 | 160 | 31pd | DefaultC | 0.9537 | -0.0021 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 92 | OFC | Limbic_OFC | +0.0032 |
| 91 | 13l | Limbic_OFC | +0.0023 |
| 165 | pOFC | Limbic_OFC | +0.0023 |
| 89 | 10pp | Limbic_OFC | +0.0020 |
| 90 | 11l | Limbic_OFC | +0.0020 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 38 | 5L | DorsAttnB | -0.0029 |
| 161 | 31a | ContA | -0.0021 |
| 160 | 31pd | DefaultC | -0.0021 |
| 24 | PSL | SalVentAttnA | -0.0020 |
| 173 | LBelt | SomMotB | -0.0020 |

### Cluster 1 — Top 10 ROIs by Importance (N=8)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 171 | TGv | Limbic_TempPole | 2.0841 | +0.1117 | +ADHD |
| 2 | 92 | OFC | Limbic_OFC | 1.9267 | +0.0039 | +ADHD |
| 3 | 177 | PI | SalVentAttnA | 1.8046 | +0.0231 | +ADHD |
| 4 | 38 | 5L | DorsAttnB | 1.7805 | +0.0011 | +ADHD |
| 5 | 89 | 10pp | Limbic_OFC | 1.7760 | +0.0055 | +ADHD |
| 6 | 165 | pOFC | Limbic_OFC | 1.7493 | +0.0131 | +ADHD |
| 7 | 161 | 31a | ContA | 1.7329 | +0.0026 | +ADHD |
| 8 | 60 | a24 | DefaultC | 1.7127 | +0.0059 | +ADHD |
| 9 | 54 | 6mp | SomMotA | 1.6502 | -0.0001 | -ADHD |
| 10 | 90 | 11l | Limbic_OFC | 1.5758 | +0.0086 | +ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.1117 |
| 121 | PeEc | Limbic_TempPole | +0.0679 |
| 134 | TF | Limbic_TempPole | +0.0665 |
| 130 | TGd | Limbic_TempPole | +0.0457 |
| 117 | EC | Limbic_TempPole | +0.0359 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 102 | 52 | SomMotB | -0.0030 |
| 69 | 8BL | TempPar | -0.0028 |
| 114 | FOP2 | SomMotB | -0.0026 |
| 123 | PBelt | SomMotB | -0.0023 |
| 99 | OP4 | SomMotB | -0.0023 |

### Cluster 2 — Top 10 ROIs by Importance (N=108)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 171 | TGv | Limbic_TempPole | 0.8054 | +0.0010 | +ADHD |
| 2 | 38 | 5L | DorsAttnB | 0.7863 | -0.0003 | -ADHD |
| 3 | 134 | TF | Limbic_TempPole | 0.7827 | -0.0013 | -ADHD |
| 4 | 73 | 44 | TempPar | 0.7706 | -0.0001 | -ADHD |
| 5 | 168 | FOP5 | SalVentAttnB | 0.7664 | +0.0004 | +ADHD |
| 6 | 20 | LO2 | VisCent | 0.7572 | -0.0009 | -ADHD |
| 7 | 160 | 31pd | DefaultC | 0.7537 | -0.0008 | -ADHD |
| 8 | 174 | A4 | SomMotB | 0.7403 | -0.0004 | -ADHD |
| 9 | 169 | p10p | ContC | 0.7192 | -0.0000 | -ADHD |
| 10 | 135 | TE2p | Limbic_TempPole | 0.7189 | -0.0006 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 171 | TGv | Limbic_TempPole | +0.0010 |
| 177 | PI | SalVentAttnA | +0.0008 |
| 131 | TE1a | TempPar | +0.0006 |
| 93 | 47s | TempPar | +0.0006 |
| 13 | RSC | ContA | +0.0006 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 134 | TF | Limbic_TempPole | -0.0013 |
| 117 | EC | Limbic_TempPole | -0.0011 |
| 142 | PGp | DorsAttnA | -0.0010 |
| 162 | VVC | VisCent | -0.0010 |
| 136 | PHT | ContB | -0.0010 |

### Cluster 3 — Top 10 ROIs by Importance (N=122)

| Rank | ROI | Region | Network | Importance | Signed | Direction |
|:----:|:---:|--------|---------|:----------:|:------:|:---------:|
| 1 | 20 | LO2 | VisCent | 0.6927 | -0.0002 | -ADHD |
| 2 | 134 | TF | Limbic_TempPole | 0.6423 | -0.0002 | -ADHD |
| 3 | 171 | TGv | Limbic_TempPole | 0.6402 | +0.0002 | +ADHD |
| 4 | 160 | 31pd | DefaultC | 0.6321 | +0.0002 | +ADHD |
| 5 | 168 | FOP5 | SalVentAttnB | 0.5869 | +0.0001 | +ADHD |
| 6 | 73 | 44 | TempPar | 0.5821 | -0.0004 | -ADHD |
| 7 | 18 | V3B | VisCent | 0.5690 | -0.0006 | -ADHD |
| 8 | 169 | p10p | ContC | 0.5690 | +0.0004 | +ADHD |
| 9 | 38 | 5L | DorsAttnB | 0.5643 | -0.0006 | -ADHD |
| 10 | 32 | v23ab | DefaultC | 0.5600 | -0.0001 | -ADHD |

**Top 5 ROIs with positive (+ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 119 | H | Limbic_TempPole | +0.0008 |
| 120 | ProS | VisPeri | +0.0005 |
| 60 | a24 | DefaultC | +0.0005 |
| 89 | 10pp | Limbic_OFC | +0.0005 |
| 169 | p10p | ContC | +0.0004 |

**Top 5 ROIs with negative (-ADHD) relationship:**

| ROI | Region | Network | Signed Score |
|:---:|--------|---------|:------------:|
| 122 | STGa | DefaultA | -0.0007 |
| 174 | A4 | SomMotB | -0.0006 |
| 21 | PIT | VisCent | -0.0006 |
| 18 | V3B | VisCent | -0.0006 |
| 38 | 5L | DorsAttnB | -0.0006 |

### Network-Level Scores per Cluster (Absolute)

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | 0.7931 | 1.3321 | 0.5709 | 0.4424 |
| ContB | 0.6630 | 0.8901 | 0.5066 | 0.3920 |
| ContC | 0.6899 | 1.0031 | 0.5436 | 0.4313 |
| DefaultA | 0.6575 | 0.9096 | 0.5588 | 0.4195 |
| DefaultB | 0.6835 | 0.9105 | 0.5522 | 0.4400 |
| DefaultC | 0.7476 | 1.1425 | 0.5429 | 0.4269 |
| DorsAttnA | 0.6145 | 0.9285 | 0.4954 | 0.4126 |
| DorsAttnB | 0.7344 | 1.0593 | 0.5181 | 0.3981 |
| Limbic_OFC | 0.7869 | 1.5288 | 0.4881 | 0.3866 |
| Limbic_TempPole | 0.7856 | 1.4161 | 0.6029 | 0.4673 |
| SalVentAttnA | 0.7142 | 1.0307 | 0.5140 | 0.3878 |
| SalVentAttnB | 0.7290 | 1.0127 | 0.5364 | 0.4072 |
| SomMotA | 0.7713 | 1.1878 | 0.5163 | 0.3680 |
| SomMotB | 0.7285 | 0.9937 | 0.5670 | 0.4222 |
| TempPar | 0.6848 | 0.9420 | 0.5274 | 0.4128 |
| VisCent | 0.5873 | 0.7849 | 0.5225 | 0.4646 |
| VisPeri | 0.6013 | 0.8165 | 0.5131 | 0.4465 |

### Network-Level Signed Scores per Cluster (Direction)

Positive = network positively associated with ADHD in this cluster. Negative = negatively associated.

| Network | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---------|:--------:|:--------:|:--------:|:--------:|
| ContA | -0.0012 (-ADHD) | +0.0018 (+ADHD) | -0.0001 (-ADHD) | -0.0004 (-ADHD) |
| ContB | -0.0009 (-ADHD) | +0.0001 (+ADHD) | -0.0003 (-ADHD) | -0.0002 (-ADHD) |
| ContC | -0.0007 (-ADHD) | +0.0024 (+ADHD) | -0.0004 (-ADHD) | +0.0000 (+ADHD) |
| DefaultA | -0.0008 (-ADHD) | +0.0052 (+ADHD) | -0.0004 (-ADHD) | -0.0003 (-ADHD) |
| DefaultB | -0.0002 (-ADHD) | +0.0031 (+ADHD) | -0.0003 (-ADHD) | -0.0000 (-ADHD) |
| DefaultC | -0.0008 (-ADHD) | +0.0013 (+ADHD) | -0.0002 (-ADHD) | +0.0000 (+ADHD) |
| DorsAttnA | -0.0005 (-ADHD) | +0.0034 (+ADHD) | -0.0004 (-ADHD) | -0.0002 (-ADHD) |
| DorsAttnB | -0.0014 (-ADHD) | -0.0001 (-ADHD) | -0.0002 (-ADHD) | -0.0002 (-ADHD) |
| Limbic_OFC | +0.0019 (+ADHD) | +0.0067 (+ADHD) | +0.0001 (+ADHD) | +0.0002 (+ADHD) |
| Limbic_TempPole | +0.0003 (+ADHD) | +0.0472 (+ADHD) | -0.0004 (-ADHD) | +0.0001 (+ADHD) |
| SalVentAttnA | -0.0008 (-ADHD) | +0.0009 (+ADHD) | +0.0002 (+ADHD) | +0.0000 (+ADHD) |
| SalVentAttnB | -0.0006 (-ADHD) | +0.0019 (+ADHD) | +0.0002 (+ADHD) | +0.0000 (+ADHD) |
| SomMotA | -0.0014 (-ADHD) | +0.0004 (+ADHD) | -0.0003 (-ADHD) | -0.0002 (-ADHD) |
| SomMotB | -0.0011 (-ADHD) | +0.0001 (+ADHD) | -0.0001 (-ADHD) | -0.0001 (-ADHD) |
| TempPar | -0.0004 (-ADHD) | +0.0038 (+ADHD) | -0.0001 (-ADHD) | -0.0000 (-ADHD) |
| VisCent | -0.0002 (-ADHD) | +0.0024 (+ADHD) | -0.0005 (-ADHD) | -0.0002 (-ADHD) |
| VisPeri | +0.0002 (+ADHD) | +0.0025 (+ADHD) | -0.0002 (-ADHD) | +0.0000 (+ADHD) |

## Caveats

1. **Model performance ceiling**: AUC ~0.58-0.62. Subtype structure from a near-chance model is exploratory.
2. **Load balancing suppresses routing differences**: Auxiliary loss encourages uniform routing, compressing circuit-level heterogeneity.
3. **Single seed**: Results from seed=2025 only. Cluster stability across seeds is not validated.
4. **Cortical ROIs only**: HCP-MMP1 180 ROIs cover cortex. Subcortical structures (caudate, pallidum) are absent.
5. **Cluster count**: K chosen a priori. Silhouette scores should guide optimal K selection.
