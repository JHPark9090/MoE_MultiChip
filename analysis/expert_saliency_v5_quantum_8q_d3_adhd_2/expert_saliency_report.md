# Per-Expert Gradient Saliency Analysis

**Model**: quantum | **Config**: adhd_2 | **Experts**: 2

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
| **Limbic_OFC** | 8 | 0.003344 | -0.000146 | -ADHD | p=0.7796 n.s. |
| **SalVentAttnB** | 8 | 0.003340 | -0.000300 | -ADHD | p=0.6986 n.s. |
| **DefaultB** | 5 | 0.003337 | -0.000377 | -ADHD | p=0.1654 n.s. |
| **DefaultA** | 5 | 0.003308 | +0.000071 | +ADHD | p=0.5961 n.s. |
| **Limbic_TempPole** | 9 | 0.003239 | +0.000074 | +ADHD | p=0.8623 n.s. |
| **SalVentAttnA** | 19 | 0.003192 | -0.000090 | -ADHD | p=0.2924 n.s. |
| **DefaultC** | 16 | 0.003003 | +0.000070 | +ADHD | p=0.5799 n.s. |
| **TempPar** | 15 | 0.002992 | -0.000120 | -ADHD | p=0.6728 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 128 | STSdp | Area_STSd_posterior | Temp | DefaultA | 0.004506 | +0.000156 | +ADHD | 0.8864 |
| 2 | 113 | FOP3 | Frontal_Opercular_Area_3 | Ins | SalVentAttnA | 0.004504 | -0.001535 | -ADHD | 0.2852 |
| 3 | 92 | OFC | Orbital_Frontal_Complex | Fr | Limbic_OFC | 0.004333 | +0.000460 | +ADHD | 0.0129* |
| 4 | 177 | PI | Para-Insular_Area | Temp | SalVentAttnA | 0.004155 | +0.000399 | +ADHD | 0.9410 |
| 5 | 43 | 6ma | Area_6m_anterior | Fr | SalVentAttnA | 0.004019 | +0.000096 | +ADHD | 0.9483 |
| 6 | 83 | 46 | Area_46 | Fr | SalVentAttnB | 0.003958 | -0.000581 | -ADHD | 0.4864 |
| 7 | 135 | TE2p | Area_TE2_posterior | Temp | Limbic_TempPole | 0.003864 | -0.000085 | -ADHD | 0.2064 |
| 8 | 149 | PGi | Area_PGi | Par | DefaultC | 0.003845 | +0.000161 | +ADHD | 0.8935 |
| 9 | 58 | a24pr | Anterior_24_prime | Fr | SalVentAttnB | 0.003826 | -0.000434 | -ADHD | 0.4941 |
| 10 | 29 | 7m | Area_7m | Par | DefaultC | 0.003820 | +0.000540 | +ADHD | 0.0699 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 93 | 47s (Area_47s) | TempPar | +0.000958 | 0.5554 |
| 150 | PGs (Area_PGs) | DefaultC | +0.000861 | 0.0575 |
| 85 | 9-46d (Area_9-46d) | SalVentAttnB | +0.000791 | 0.3549 |
| 134 | TF (Area_TF) | Limbic_TempPole | +0.000661 | 0.6144 |
| 27 | STV (Superior_Temporal_Visual_Area) | DefaultA | +0.000582 | 0.1889 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 113 | FOP3 (Frontal_Opercular_Area_3) | SalVentAttnA | -0.001535 | 0.2852 |
| 178 | a32pr (Area_anterior_32_prime) | SalVentAttnB | -0.000977 | 0.3637 |
| 154 | PHA2 (ParaHippocampal_Area_2) | DefaultB | -0.000767 | 0.0254* |
| 131 | TE1a (Area_TE1_anterior) | TempPar | -0.000612 | 0.1188 |
| 56 | p24pr (Area_Posterior_24_prime) | SalVentAttnA | -0.000586 | 0.1572 |

---

## Expert: External (95 ROIs)

Networks: ContA, ContB, ContC, DorsAttnA, DorsAttnB, VisCent, VisPeri, SomMotA, SomMotB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the External expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **DorsAttnA** | 16 | 0.001852 | -0.000120 | -ADHD | p=0.9857 n.s. |
| **ContC** | 11 | 0.001847 | -0.000017 | -ADHD | p=0.0873 n.s. |
| **DorsAttnB** | 8 | 0.001816 | -0.000201 | -ADHD | p=0.9520 n.s. |
| **ContA** | 5 | 0.001816 | -0.000168 | -ADHD | p=0.0391 * |
| **SomMotB** | 14 | 0.001770 | -0.000129 | -ADHD | p=0.0997 n.s. |
| **VisCent** | 12 | 0.001761 | -0.000152 | -ADHD | p=0.2881 n.s. |
| **ContB** | 12 | 0.001703 | -0.000134 | -ADHD | p=0.3513 n.s. |
| **SomMotA** | 9 | 0.001652 | -0.000108 | -ADHD | p=0.3060 n.s. |
| **VisPeri** | 8 | 0.001515 | -0.000110 | -ADHD | p=0.4134 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 106 | TA2 | Area_TA2 | Temp | SomMotB | 0.002627 | -0.000510 | -ADHD | 0.3533 |
| 2 | 45 | 7Pl | Lateral_Area_7P | Par | DorsAttnA | 0.002245 | -0.000421 | -ADHD | 0.7021 |
| 3 | 174 | A4 | Auditory_4_Complex | Temp | SomMotB | 0.002215 | +0.000406 | +ADHD | 0.1087 |
| 4 | 161 | 31a | Area_31a | Par | ContA | 0.002194 | -0.000110 | -ADHD | 0.1092 |
| 5 | 21 | PIT | Posterior_InferoTemporal_complex | Occ | VisCent | 0.002189 | +0.000111 | +ADHD | 0.6396 |
| 6 | 44 | 7Am | Medial_Area_7A | Par | DorsAttnB | 0.002181 | -0.000314 | -ADHD | 0.9468 |
| 7 | 100 | OP1 | Area_OP1-SII | Par | SomMotB | 0.002173 | +0.000054 | +ADHD | 0.2512 |
| 8 | 17 | FFC | Fusiform_Face_Complex | Temp | DorsAttnA | 0.002153 | +0.000218 | +ADHD | 0.7466 |
| 9 | 148 | PFm | Area_PFm_Complex | Par | ContC | 0.002143 | +0.000054 | +ADHD | 0.7117 |
| 10 | 49 | MIP | Medial_IntraParietal_Area | Par | DorsAttnA | 0.002100 | -0.000408 | -ADHD | 0.8514 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 174 | A4 (Auditory_4_Complex) | SomMotB | +0.000406 | 0.1087 |
| 132 | TE1p (Area_TE1_posterior) | ContC | +0.000254 | 0.0515 |
| 96 | i6-8 (Inferior_6-8_Transitional_Area) | ContC | +0.000234 | 0.0342* |
| 17 | FFC (Fusiform_Face_Complex) | DorsAttnA | +0.000218 | 0.7466 |
| 7 | 4 (Primary_Motor_Cortex) | SomMotA | +0.000204 | 0.3455 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 106 | TA2 (Area_TA2) | SomMotB | -0.000510 | 0.3533 |
| 20 | LO2 (Area_Lateral_Occipital_2) | VisCent | -0.000479 | 0.1780 |
| 45 | 7Pl (Lateral_Area_7P) | DorsAttnA | -0.000421 | 0.7021 |
| 157 | V3CD (Area_V3CD) | VisCent | -0.000417 | 0.1268 |
| 49 | MIP (Medial_IntraParietal_Area) | DorsAttnA | -0.000408 | 0.8514 |

---

## Cross-Expert Comparison

### Dominant Network Per Expert

Which sub-network does each expert prioritize most?

| Expert | Top Network (abs) | Abs Saliency | Top Network (signed) | Signed | Direction |
|--------|:-----------------:|:------------:|:--------------------:|:------:|:---------:|
| Internal | Limbic_OFC | 0.003344 | DefaultB | -0.000377 | -ADHD |
| External | DorsAttnA | 0.001852 | DorsAttnB | -0.000201 | -ADHD |

### Intra-Circuit Feature Hierarchy (All Experts)

Networks ranked by absolute saliency within each expert. This reveals whether experts prioritize their 'expected' sub-networks or learn unexpected feature hierarchies.

- **Internal**: Limbic_OFC (0.0033, -ADHD) > SalVentAttnB (0.0033, -ADHD) > DefaultB (0.0033, -ADHD) > DefaultA (0.0033, +ADHD) > Limbic_TempPole (0.0032, +ADHD) > SalVentAttnA (0.0032, -ADHD) > DefaultC (0.0030, +ADHD) > TempPar (0.0030, -ADHD)
- **External**: DorsAttnA (0.0019, -ADHD) > ContC (0.0018, -ADHD) > DorsAttnB (0.0018, -ADHD) > ContA (0.0018, -ADHD) > SomMotB (0.0018, -ADHD) > VisCent (0.0018, -ADHD) > ContB (0.0017, -ADHD) > SomMotA (0.0017, -ADHD) > VisPeri (0.0015, -ADHD)

---

## Significant Findings (p < 0.05)

- **Internal expert → ROI p24** (DefaultC): p=0.0002, +ADHD, signed=+0.000570
- **Internal expert → ROI PFcm** (SalVentAttnA): p=0.0099, +ADHD, signed=+0.000545
- **Internal expert → ROI PHA3** (DefaultB): p=0.0107, -ADHD, signed=-0.000229
- **External expert → ContA**: Signed saliency differs between ADHD+ and ADHD- (p=0.0391, direction=-ADHD)
- **External expert → ROI PBelt** (SomMotB): p=0.0076, -ADHD, signed=-0.000344
- **External expert → ROI PHT** (ContB): p=0.0288, +ADHD, signed=+0.000071
- **External expert → ROI Ig** (SomMotB): p=0.0313, -ADHD, signed=-0.000080
