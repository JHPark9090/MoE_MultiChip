# Per-Expert Gradient Saliency Analysis

**Model**: classical | **Config**: adhd_2 | **Experts**: 2

## Methodology

For each expert, we compute ∂(expert_k output)/∂(expert_k input ROIs), isolating what each expert has learned to attend to within its assigned brain circuit. This is distinct from model-level saliency (∂logits/∂input), which mixes contributions from all experts and gating.

- **Absolute saliency**: Magnitude of gradient (how much this ROI matters to this expert, regardless of direction)
- **Signed saliency**: Direction of gradient (+ADHD = increasing activity pushes expert output in ADHD-associated direction)
- **Statistical tests**: Welch's t-test comparing ADHD+ vs ADHD- subjects at network and ROI levels

---

## Expert: Internal (85 ROIs)

Networks: DefaultA, DefaultB, DefaultC, TempPar, Limbic_TempPole, Limbic_OFC, SalVentAttnA, SalVentAttnB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the Internal expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **Limbic_TempPole** | 9 | 0.000863 | -0.000029 | -ADHD | p=0.7087 n.s. |
| **DefaultC** | 16 | 0.000720 | +0.000057 | +ADHD | p=0.9696 n.s. |
| **SalVentAttnB** | 8 | 0.000700 | +0.000050 | +ADHD | p=0.5525 n.s. |
| **Limbic_OFC** | 8 | 0.000663 | -0.000059 | -ADHD | p=0.1712 n.s. |
| **SalVentAttnA** | 19 | 0.000658 | -0.000014 | -ADHD | p=0.9421 n.s. |
| **DefaultB** | 5 | 0.000632 | -0.000051 | -ADHD | p=0.9581 n.s. |
| **TempPar** | 15 | 0.000621 | -0.000009 | -ADHD | p=0.1659 n.s. |
| **DefaultA** | 5 | 0.000611 | -0.000066 | -ADHD | p=0.4003 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 171 | TGv | Area_TG_Ventral | Temp | Limbic_TempPole | 0.001649 | -0.000277 | -ADHD | 0.9100 |
| 2 | 160 | 31pd | Area_31pd | Par | DefaultC | 0.001348 | -0.000296 | -ADHD | 0.4530 |
| 3 | 73 | 44 | Area_44 | Fr | TempPar | 0.001330 | +0.000029 | +ADHD | 0.3189 |
| 4 | 168 | FOP5 | Area_Frontal_Opercular_5 | Fr | SalVentAttnB | 0.001250 | +0.000070 | +ADHD | 0.8163 |
| 5 | 134 | TF | Area_TF | Temp | Limbic_TempPole | 0.001236 | -0.000119 | -ADHD | 0.9954 |
| 6 | 43 | 6ma | Area_6m_anterior | Fr | SalVentAttnA | 0.001180 | +0.000154 | +ADHD | 0.4409 |
| 7 | 68 | 9m | Area_9_Middle | Fr | TempPar | 0.001174 | -0.000172 | -ADHD | 0.6908 |
| 8 | 135 | TE2p | Area_TE2_posterior | Temp | Limbic_TempPole | 0.001162 | +0.000129 | +ADHD | 0.9626 |
| 9 | 149 | PGi | Area_PGi | Par | DefaultC | 0.001111 | +0.000182 | +ADHD | 0.9007 |
| 10 | 89 | 10pp | Polar_10p | Fr | Limbic_OFC | 0.001099 | -0.000085 | -ADHD | 0.3744 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 60 | a24 (Area_a24) | DefaultC | +0.000274 | 0.7057 |
| 179 | p24 (Area_posterior_24) | DefaultC | +0.000204 | 0.5021 |
| 149 | PGi (Area_PGi) | DefaultC | +0.000182 | 0.9007 |
| 175 | STSva (Area_STSv_anterior) | TempPar | +0.000164 | 0.5937 |
| 43 | 6ma (Area_6m_anterior) | SalVentAttnA | +0.000154 | 0.4409 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 160 | 31pd (Area_31pd) | DefaultC | -0.000296 | 0.4530 |
| 171 | TGv (Area_TG_Ventral) | Limbic_TempPole | -0.000277 | 0.9100 |
| 24 | PSL (PeriSylvian_Language_Area) | SalVentAttnA | -0.000201 | 0.7974 |
| 92 | OFC (Orbital_Frontal_Complex) | Limbic_OFC | -0.000201 | 0.5579 |
| 128 | STSdp (Area_STSd_posterior) | DefaultA | -0.000173 | 0.5646 |

---

## Expert: External (95 ROIs)

Networks: ContA, ContB, ContC, DorsAttnA, DorsAttnB, VisCent, VisPeri, SomMotA, SomMotB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the External expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **ContA** | 5 | 0.000375 | +0.000060 | +ADHD | p=0.0365 * |
| **DorsAttnB** | 8 | 0.000348 | +0.000020 | +ADHD | p=0.8296 n.s. |
| **VisCent** | 12 | 0.000302 | -0.000054 | -ADHD | p=0.0333 * |
| **SomMotB** | 14 | 0.000300 | -0.000001 | -ADHD | p=0.0883 n.s. |
| **SomMotA** | 9 | 0.000298 | +0.000052 | +ADHD | p=0.0231 * |
| **ContC** | 11 | 0.000298 | -0.000030 | -ADHD | p=0.9691 n.s. |
| **VisPeri** | 8 | 0.000260 | -0.000051 | -ADHD | p=0.2812 n.s. |
| **DorsAttnA** | 16 | 0.000257 | -0.000010 | -ADHD | p=0.4812 n.s. |
| **ContB** | 12 | 0.000254 | -0.000004 | -ADHD | p=0.1180 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 38 | 5L | Area_5L | Par | DorsAttnB | 0.000689 | +0.000077 | +ADHD | 0.8630 |
| 2 | 161 | 31a | Area_31a | Par | ContA | 0.000657 | +0.000196 | +ADHD | 0.0550 |
| 3 | 54 | 6mp | Area_6mp | Fr | SomMotA | 0.000647 | +0.000034 | +ADHD | 0.2251 |
| 4 | 174 | A4 | Auditory_4_Complex | Temp | SomMotB | 0.000554 | -0.000023 | -ADHD | 0.1551 |
| 5 | 66 | 8Av | Area_8Av | Fr | ContC | 0.000496 | +0.000042 | +ADHD | 0.1949 |
| 6 | 155 | V4t | Area_V4t | Occ | VisCent | 0.000487 | -0.000057 | -ADHD | 0.1236 |
| 7 | 99 | OP4 | Area_OP4-PV | Par | SomMotB | 0.000486 | +0.000004 | +ADHD | 0.4614 |
| 8 | 144 | IP1 | Area_IntraParietal_1 | Par | ContB | 0.000455 | +0.000034 | +ADHD | 0.1483 |
| 9 | 18 | V3B | Area_V3B | Occ | VisCent | 0.000455 | -0.000097 | -ADHD | 0.1022 |
| 10 | 20 | LO2 | Area_Lateral_Occipital_2 | Occ | VisCent | 0.000450 | -0.000201 | -ADHD | 0.3453 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 161 | 31a (Area_31a) | ContA | +0.000196 | 0.0550 |
| 94 | LIPd (Area_Lateral_IntraParietal_dorsal) | ContB | +0.000127 | 0.4347 |
| 136 | PHT (Area_PHT) | ContB | +0.000119 | 0.1010 |
| 123 | PBelt (ParaBelt_Complex) | SomMotB | +0.000115 | 0.7612 |
| 53 | 6d (Dorsal_area_6) | SomMotA | +0.000109 | 0.0316* |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 20 | LO2 (Area_Lateral_Occipital_2) | VisCent | -0.000201 | 0.3453 |
| 2 | V6 (Sixth_Visual_Area) | VisPeri | -0.000159 | 0.3265 |
| 169 | p10p (Area_posterior_10p) | ContC | -0.000157 | 0.3390 |
| 76 | a47r (Area_anterior_47r) | ContC | -0.000114 | 0.1855 |
| 81 | IFSa (Area_IFSa) | ContB | -0.000109 | 0.0264* |

---

## Cross-Expert Comparison

### Dominant Network Per Expert

Which sub-network does each expert prioritize most?

| Expert | Top Network (abs) | Abs Saliency | Top Network (signed) | Signed | Direction |
|--------|:-----------------:|:------------:|:--------------------:|:------:|:---------:|
| Internal | Limbic_TempPole | 0.000863 | DefaultA | -0.000066 | -ADHD |
| External | ContA | 0.000375 | ContA | +0.000060 | +ADHD |

### Intra-Circuit Feature Hierarchy (All Experts)

Networks ranked by absolute saliency within each expert. This reveals whether experts prioritize their 'expected' sub-networks or learn unexpected feature hierarchies.

- **Internal**: Limbic_TempPole (0.0009, -ADHD) > DefaultC (0.0007, +ADHD) > SalVentAttnB (0.0007, +ADHD) > Limbic_OFC (0.0007, -ADHD) > SalVentAttnA (0.0007, -ADHD) > DefaultB (0.0006, -ADHD) > TempPar (0.0006, -ADHD) > DefaultA (0.0006, -ADHD)
- **External**: ContA (0.0004, +ADHD) > DorsAttnB (0.0003, +ADHD) > VisCent (0.0003, -ADHD) > SomMotB (0.0003, -ADHD) > SomMotA (0.0003, +ADHD) > ContC (0.0003, -ADHD) > VisPeri (0.0003, -ADHD) > DorsAttnA (0.0003, -ADHD) > ContB (0.0003, -ADHD)

---

## Significant Findings (p < 0.05)

- **External expert → ContA**: Signed saliency differs between ADHD+ and ADHD- (p=0.0365, direction=+ADHD)
- **External expert → VisCent**: Signed saliency differs between ADHD+ and ADHD- (p=0.0333, direction=-ADHD)
- **External expert → SomMotA**: Signed saliency differs between ADHD+ and ADHD- (p=0.0231, direction=+ADHD)
- **External expert → ROI Ig** (SomMotB): p=0.0051, -ADHD, signed=-0.000093
- **External expert → ROI 4** (SomMotA): p=0.0109, +ADHD, signed=+0.000086
- **External expert → ROI VMV3** (VisCent): p=0.0148, -ADHD, signed=-0.000061
