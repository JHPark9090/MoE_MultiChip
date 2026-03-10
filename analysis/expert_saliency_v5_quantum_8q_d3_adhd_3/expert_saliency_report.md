# Per-Expert Gradient Saliency Analysis

**Model**: quantum | **Config**: adhd_3 | **Experts**: 4

## Methodology

For each expert, we compute ∂(expert_k output)/∂(expert_k input ROIs), isolating what each expert has learned to attend to within its assigned brain circuit. This is distinct from model-level saliency (∂logits/∂input), which mixes contributions from all experts and gating.

- **Absolute saliency**: Magnitude of gradient (how much this ROI matters to this expert, regardless of direction)
- **Signed saliency**: Direction of gradient (+ADHD = increasing activity pushes expert output in ADHD-associated direction)
- **Statistical tests**: Welch's t-test comparing ADHD+ vs ADHD- subjects at network and ROI levels

---

## Expert: DMN (41 ROIs)

Networks: DefaultA, DefaultB, DefaultC, TempPar

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the DMN expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **DefaultC** | 16 | 0.003953 | +0.000335 | +ADHD | p=0.9865 n.s. |
| **DefaultB** | 5 | 0.003837 | -0.000222 | -ADHD | p=0.8024 n.s. |
| **TempPar** | 15 | 0.003741 | -0.000118 | -ADHD | p=0.9350 n.s. |
| **DefaultA** | 5 | 0.003565 | -0.000079 | -ADHD | p=0.9433 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 175 | STSva | Area_STSv_anterior | Temp | TempPar | 0.005119 | +0.001649 | +ADHD | 0.0727 |
| 2 | 32 | v23ab | Area_ventral_23_a+b | Par | DefaultC | 0.004541 | +0.000306 | +ADHD | 0.6916 |
| 3 | 70 | 9p | Area_9_Posterior | Fr | TempPar | 0.004488 | -0.000244 | -ADHD | 0.1479 |
| 4 | 129 | STSvp | Area_STSv_posterior | Temp | TempPar | 0.004479 | +0.000243 | +ADHD | 0.4770 |
| 5 | 126 | PHA3 | ParaHippocampal_Area_3 | Temp | DefaultB | 0.004438 | +0.000966 | +ADHD | 0.0815 |
| 6 | 128 | STSdp | Area_STSd_posterior | Temp | DefaultA | 0.004311 | +0.000950 | +ADHD | 0.0626 |
| 7 | 125 | PHA1 | ParaHippocampal_Area_1 | Temp | DefaultB | 0.004273 | -0.001165 | -ADHD | 0.2583 |
| 8 | 61 | d32 | Area_dorsal_32 | Fr | DefaultC | 0.004270 | +0.000507 | +ADHD | 0.9521 |
| 9 | 67 | 8Ad | Area_8Ad | Fr | DefaultC | 0.004262 | +0.000353 | +ADHD | 0.6693 |
| 10 | 149 | PGi | Area_PGi | Par | DefaultC | 0.004256 | +0.000838 | +ADHD | 0.0055* |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 64 | 10r (Area_10r) | DefaultC | +0.001715 | 0.0292* |
| 175 | STSva (Area_STSv_anterior) | TempPar | +0.001649 | 0.0727 |
| 60 | a24 (Area_a24) | DefaultC | +0.001440 | 0.6837 |
| 29 | 7m (Area_7m) | DefaultC | +0.001013 | 0.5694 |
| 126 | PHA3 (ParaHippocampal_Area_3) | DefaultB | +0.000966 | 0.0815 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 34 | 31pv (Area_31p_ventral) | DefaultC | -0.001492 | 0.7857 |
| 65 | 47m (Area_47m) | TempPar | -0.001222 | 0.9423 |
| 125 | PHA1 (ParaHippocampal_Area_1) | DefaultB | -0.001165 | 0.2583 |
| 73 | 44 (Area_44) | TempPar | -0.000958 | 0.2527 |
| 131 | TE1a (Area_TE1_anterior) | TempPar | -0.000927 | 0.7120 |

---

## Expert: Executive (52 ROIs)

Networks: ContA, ContB, ContC, DorsAttnA, DorsAttnB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the Executive expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **ContB** | 12 | 0.003683 | -0.000049 | -ADHD | p=0.2634 n.s. |
| **ContC** | 11 | 0.003547 | +0.000241 | +ADHD | p=0.5401 n.s. |
| **DorsAttnA** | 16 | 0.003544 | -0.000175 | -ADHD | p=0.7699 n.s. |
| **DorsAttnB** | 8 | 0.003502 | -0.000478 | -ADHD | p=0.5620 n.s. |
| **ContA** | 5 | 0.003357 | -0.000067 | -ADHD | p=0.7939 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 88 | a10p | Area_anterior_10p | Fr | ContC | 0.004502 | +0.000525 | +ADHD | 0.8667 |
| 2 | 156 | FST | Area_FST | Occ | DorsAttnA | 0.004385 | -0.001321 | -ADHD | 0.5439 |
| 3 | 44 | 7Am | Medial_Area_7A | Par | DorsAttnB | 0.004357 | -0.000821 | -ADHD | 0.7606 |
| 4 | 82 | p9-46v | Area_posterior_9-46v | Fr | ContB | 0.004204 | -0.000928 | -ADHD | 0.5902 |
| 5 | 80 | IFSp | Area_IFSp | Fr | ContB | 0.004165 | -0.000632 | -ADHD | 0.7288 |
| 6 | 143 | IP2 | Area_IntraParietal_2 | Par | ContB | 0.004094 | -0.000912 | -ADHD | 0.2390 |
| 7 | 148 | PFm | Area_PFm_Complex | Par | ContC | 0.004048 | -0.000209 | -ADHD | 0.6259 |
| 8 | 17 | FFC | Fusiform_Face_Complex | Temp | DorsAttnA | 0.004023 | -0.000661 | -ADHD | 0.5367 |
| 9 | 79 | IFJp | Area_IFJp | Fr | ContB | 0.003973 | +0.000250 | +ADHD | 0.9574 |
| 10 | 45 | 7Pl | Lateral_Area_7P | Par | DorsAttnA | 0.003972 | -0.001138 | -ADHD | 0.8835 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 84 | a9-46v (Area_anterior_9-46v) | ContC | +0.001044 | 0.9157 |
| 72 | 8C (Area_8C) | ContC | +0.001024 | 0.0299* |
| 158 | LO3 (Area_Lateral_Occipital_3) | DorsAttnA | +0.000769 | 0.1594 |
| 76 | a47r (Area_anterior_47r) | ContC | +0.000593 | 0.0208* |
| 13 | RSC (RetroSplenial_Complex) | ContA | +0.000558 | 0.2896 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 156 | FST (Area_FST) | DorsAttnA | -0.001321 | 0.5439 |
| 45 | 7Pl (Lateral_Area_7P) | DorsAttnA | -0.001138 | 0.8835 |
| 115 | PFt (Area_PFt) | DorsAttnB | -0.000971 | 0.2500 |
| 82 | p9-46v (Area_posterior_9-46v) | ContB | -0.000928 | 0.5902 |
| 143 | IP2 (Area_IntraParietal_2) | ContB | -0.000912 | 0.2390 |

---

## Expert: Salience (44 ROIs)

Networks: SalVentAttnA, SalVentAttnB, Limbic_TempPole, Limbic_OFC

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the Salience expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **Limbic_TempPole** | 9 | 0.005402 | -0.000179 | -ADHD | p=0.4056 n.s. |
| **Limbic_OFC** | 8 | 0.005236 | -0.000100 | -ADHD | p=0.1072 n.s. |
| **SalVentAttnB** | 8 | 0.004989 | +0.000676 | +ADHD | p=0.2206 n.s. |
| **SalVentAttnA** | 19 | 0.004948 | +0.000255 | +ADHD | p=0.1615 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 134 | TF | Area_TF | Temp | Limbic_TempPole | 0.008089 | -0.001147 | -ADHD | 0.5958 |
| 2 | 171 | TGv | Area_TG_Ventral | Temp | Limbic_TempPole | 0.007488 | -0.001116 | -ADHD | 0.1898 |
| 3 | 92 | OFC | Orbital_Frontal_Complex | Fr | Limbic_OFC | 0.006838 | +0.000726 | +ADHD | 0.5727 |
| 4 | 177 | PI | Para-Insular_Area | Temp | SalVentAttnA | 0.006144 | +0.002655 | +ADHD | 0.1700 |
| 5 | 110 | AVI | Anterior_Ventral_Insular_Area | Fr | SalVentAttnB | 0.005934 | +0.003149 | +ADHD | 0.0541 |
| 6 | 163 | 25 | Area_25 | Fr | Limbic_OFC | 0.005883 | +0.000970 | +ADHD | 0.7403 |
| 7 | 178 | a32pr | Area_anterior_32_prime | Fr | SalVentAttnB | 0.005734 | -0.000083 | -ADHD | 0.1279 |
| 8 | 56 | p24pr | Area_Posterior_24_prime | Fr | SalVentAttnA | 0.005604 | -0.001281 | -ADHD | 0.4636 |
| 9 | 164 | s32 | Area_s32 | Fr | Limbic_OFC | 0.005598 | -0.000483 | -ADHD | 0.0050* |
| 10 | 40 | 24dv | Ventral_Area_24d | Fr | SalVentAttnA | 0.005537 | +0.000758 | +ADHD | 0.2910 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 110 | AVI (Anterior_Ventral_Insular_Area) | SalVentAttnB | +0.003149 | 0.0541 |
| 37 | 23c (Area_23c) | SalVentAttnA | +0.002694 | 0.7874 |
| 177 | PI (Para-Insular_Area) | SalVentAttnA | +0.002655 | 0.1700 |
| 85 | 9-46d (Area_9-46d) | SalVentAttnB | +0.002134 | 0.0006* |
| 112 | FOP1 (Frontal_Opercular_Area_1) | SalVentAttnA | +0.001933 | 0.3783 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 59 | p32pr (Area_p32_prime) | SalVentAttnA | -0.002121 | 0.0143* |
| 147 | PF (Area_PF_Complex) | SalVentAttnA | -0.001928 | 0.0184* |
| 56 | p24pr (Area_Posterior_24_prime) | SalVentAttnA | -0.001281 | 0.4636 |
| 166 | PoI1 (Area_Posterior_Insular_1) | SalVentAttnA | -0.001207 | 0.2152 |
| 134 | TF (Area_TF) | Limbic_TempPole | -0.001147 | 0.5958 |

---

## Expert: SensoriMotor (43 ROIs)

Networks: VisCent, VisPeri, SomMotA, SomMotB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the SensoriMotor expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **SomMotB** | 14 | 0.004364 | +0.000128 | +ADHD | p=0.3701 n.s. |
| **SomMotA** | 9 | 0.004339 | +0.000303 | +ADHD | p=0.3614 n.s. |
| **VisCent** | 12 | 0.004092 | +0.000509 | +ADHD | p=0.3504 n.s. |
| **VisPeri** | 8 | 0.004042 | +0.000368 | +ADHD | p=0.0114 * |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 23 | A1 | Primary_Auditory_Cortex | Temp | SomMotB | 0.005431 | +0.000883 | +ADHD | 0.2399 |
| 2 | 4 | V3 | Third_Visual_Area | Occ | VisCent | 0.004812 | +0.000227 | +ADHD | 0.6244 |
| 3 | 21 | PIT | Posterior_InferoTemporal_complex | Occ | VisCent | 0.004804 | +0.001639 | +ADHD | 0.8423 |
| 4 | 106 | TA2 | Area_TA2 | Temp | SomMotB | 0.004756 | +0.001502 | +ADHD | 0.9396 |
| 5 | 8 | 3b | Primary_Sensory_Cortex | Par | SomMotA | 0.004643 | -0.001682 | -ADHD | 0.3432 |
| 6 | 53 | 6d | Dorsal_area_6 | Fr | SomMotA | 0.004640 | +0.000959 | +ADHD | 0.1019 |
| 7 | 172 | MBelt | Medial_Belt_Complex | Temp | SomMotB | 0.004610 | -0.001235 | -ADHD | 0.9078 |
| 8 | 39 | 24dd | Dorsal_Area_24d | Fr | SomMotA | 0.004566 | +0.000654 | +ADHD | 0.1819 |
| 9 | 123 | PBelt | ParaBelt_Complex | Temp | SomMotB | 0.004463 | -0.000713 | -ADHD | 0.6389 |
| 10 | 35 | 5m | Area_5m | Par | SomMotA | 0.004454 | +0.000264 | +ADHD | 0.6278 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 101 | OP2-3 (Area_OP2-3-VS) | SomMotB | +0.001674 | 0.3769 |
| 21 | PIT (Posterior_InferoTemporal_complex) | VisCent | +0.001639 | 0.8423 |
| 106 | TA2 (Area_TA2) | SomMotB | +0.001502 | 0.9396 |
| 15 | V7 (Seventh_Visual_Area) | VisCent | +0.001491 | 0.4816 |
| 0 | V1 (Primary_Visual_Cortex) | VisPeri | +0.001312 | 0.0069* |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 8 | 3b (Primary_Sensory_Cortex) | SomMotA | -0.001682 | 0.3432 |
| 172 | MBelt (Medial_Belt_Complex) | SomMotB | -0.001235 | 0.9078 |
| 3 | V2 (Second_Visual_Area) | VisPeri | -0.001109 | 0.8910 |
| 173 | LBelt (Lateral_Belt_Complex) | SomMotB | -0.001088 | 0.2234 |
| 100 | OP1 (Area_OP1-SII) | SomMotB | -0.000717 | 0.3048 |

---

## Cross-Expert Comparison

### Dominant Network Per Expert

Which sub-network does each expert prioritize most?

| Expert | Top Network (abs) | Abs Saliency | Top Network (signed) | Signed | Direction |
|--------|:-----------------:|:------------:|:--------------------:|:------:|:---------:|
| DMN | DefaultC | 0.003953 | DefaultC | +0.000335 | +ADHD |
| Executive | ContB | 0.003683 | DorsAttnB | -0.000478 | -ADHD |
| Salience | Limbic_TempPole | 0.005402 | SalVentAttnB | +0.000676 | +ADHD |
| SensoriMotor | SomMotB | 0.004364 | VisCent | +0.000509 | +ADHD |

### Intra-Circuit Feature Hierarchy (All Experts)

Networks ranked by absolute saliency within each expert. This reveals whether experts prioritize their 'expected' sub-networks or learn unexpected feature hierarchies.

- **DMN**: DefaultC (0.0040, +ADHD) > DefaultB (0.0038, -ADHD) > TempPar (0.0037, -ADHD) > DefaultA (0.0036, -ADHD)
- **Executive**: ContB (0.0037, -ADHD) > ContC (0.0035, +ADHD) > DorsAttnA (0.0035, -ADHD) > DorsAttnB (0.0035, -ADHD) > ContA (0.0034, -ADHD)
- **Salience**: Limbic_TempPole (0.0054, -ADHD) > Limbic_OFC (0.0052, -ADHD) > SalVentAttnB (0.0050, +ADHD) > SalVentAttnA (0.0049, +ADHD)
- **SensoriMotor**: SomMotB (0.0044, +ADHD) > SomMotA (0.0043, +ADHD) > VisCent (0.0041, +ADHD) > VisPeri (0.0040, +ADHD)

---

## Significant Findings (p < 0.05)

- **DMN expert → ROI PGi** (DefaultC): p=0.0055, +ADHD, signed=+0.000838
- **DMN expert → ROI 10r** (DefaultC): p=0.0292, +ADHD, signed=+0.001715
- **DMN expert → ROI s6-8** (DefaultC): p=0.0339, -ADHD, signed=-0.000755
- **Executive expert → ROI a47r** (ContC): p=0.0208, +ADHD, signed=+0.000593
- **Executive expert → ROI 8BM** (ContC): p=0.0290, -ADHD, signed=-0.000226
- **Executive expert → ROI 8C** (ContC): p=0.0299, +ADHD, signed=+0.001024
- **Salience expert → ROI 9-46d** (SalVentAttnB): p=0.0006, +ADHD, signed=+0.002134
- **Salience expert → ROI s32** (Limbic_OFC): p=0.0050, -ADHD, signed=-0.000483
- **Salience expert → ROI 5mv** (SalVentAttnA): p=0.0077, -ADHD, signed=-0.000737
- **SensoriMotor expert → VisPeri**: Signed saliency differs between ADHD+ and ADHD- (p=0.0114, direction=+ADHD)
- **SensoriMotor expert → ROI VMV2** (VisPeri): p=0.0049, +ADHD, signed=+0.000612
- **SensoriMotor expert → ROI RI** (SomMotB): p=0.0058, -ADHD, signed=-0.000488
- **SensoriMotor expert → ROI V1** (VisPeri): p=0.0069, +ADHD, signed=+0.001312
