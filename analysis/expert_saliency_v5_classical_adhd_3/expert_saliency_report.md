# Per-Expert Gradient Saliency Analysis

**Model**: classical | **Config**: adhd_3 | **Experts**: 4

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
| **DefaultA** | 5 | 0.000032 | -0.000002 | -ADHD | p=0.0537 n.s. |
| **DefaultB** | 5 | 0.000025 | +0.000020 | +ADHD | p=0.0117 * |
| **DefaultC** | 16 | 0.000024 | +0.000011 | +ADHD | p=0.0020 ** |
| **TempPar** | 15 | 0.000023 | -0.000011 | -ADHD | p=0.0104 * |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 63 | p32 | Area_p32 | Fr | DefaultC | 0.000063 | +0.000063 | +ADHD | 0.0012* |
| 2 | 31 | 23d | Area_23d | Par | DefaultC | 0.000057 | +0.000057 | +ADHD | 0.0024* |
| 3 | 128 | STSdp | Area_STSd_posterior | Temp | DefaultA | 0.000055 | -0.000055 | -ADHD | 0.0008* |
| 4 | 27 | STV | Superior_Temporal_Visual_Area | Par | DefaultA | 0.000050 | +0.000050 | +ADHD | 0.0030* |
| 5 | 154 | PHA2 | ParaHippocampal_Area_2 | Temp | DefaultB | 0.000049 | +0.000049 | +ADHD | 0.0004* |
| 6 | 97 | s6-8 | Superior_6-8_Transitional_Area | Fr | DefaultC | 0.000047 | +0.000047 | +ADHD | 0.0049* |
| 7 | 179 | p24 | Area_posterior_24 | Fr | DefaultC | 0.000042 | -0.000042 | -ADHD | 0.0031* |
| 8 | 127 | STSda | Area_STSd_anterior | Temp | TempPar | 0.000042 | -0.000042 | -ADHD | 0.0538 |
| 9 | 70 | 9p | Area_9_Posterior | Fr | TempPar | 0.000040 | -0.000040 | -ADHD | 0.0014* |
| 10 | 93 | 47s | Area_47s | Fr | TempPar | 0.000040 | -0.000040 | -ADHD | 0.0026* |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 63 | p32 (Area_p32) | DefaultC | +0.000063 | 0.0012* |
| 31 | 23d (Area_23d) | DefaultC | +0.000057 | 0.0024* |
| 27 | STV (Superior_Temporal_Visual_Area) | DefaultA | +0.000050 | 0.0030* |
| 154 | PHA2 (ParaHippocampal_Area_2) | DefaultB | +0.000049 | 0.0004* |
| 97 | s6-8 (Superior_6-8_Transitional_Area) | DefaultC | +0.000047 | 0.0049* |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 128 | STSdp (Area_STSd_posterior) | DefaultA | -0.000055 | 0.0008* |
| 179 | p24 (Area_posterior_24) | DefaultC | -0.000042 | 0.0031* |
| 127 | STSda (Area_STSd_anterior) | TempPar | -0.000042 | 0.0538 |
| 70 | 9p (Area_9_Posterior) | TempPar | -0.000040 | 0.0014* |
| 93 | 47s (Area_47s) | TempPar | -0.000040 | 0.0026* |

---

## Expert: Executive (52 ROIs)

Networks: ContA, ContB, ContC, DorsAttnA, DorsAttnB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the Executive expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **ContC** | 11 | 0.000024 | -0.000003 | -ADHD | p=0.0007 *** |
| **ContB** | 12 | 0.000021 | -0.000006 | -ADHD | p=0.0298 * |
| **DorsAttnB** | 8 | 0.000016 | -0.000007 | -ADHD | p=0.8608 n.s. |
| **DorsAttnA** | 16 | 0.000015 | +0.000001 | +ADHD | p=0.0419 * |
| **ContA** | 5 | 0.000012 | -0.000007 | -ADHD | p=0.0561 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 170 | p47r | Area_posterior_47r | Fr | ContC | 0.000060 | -0.000060 | -ADHD | 0.0014* |
| 2 | 72 | 8C | Area_8C | Fr | ContC | 0.000044 | +0.000044 | +ADHD | 0.0057* |
| 3 | 144 | IP1 | Area_IntraParietal_1 | Par | ContB | 0.000036 | +0.000036 | +ADHD | 0.0021* |
| 4 | 136 | PHT | Area_PHT | Temp | ContB | 0.000035 | -0.000035 | -ADHD | 0.0006* |
| 5 | 115 | PFt | Area_PFt | Par | DorsAttnB | 0.000035 | -0.000035 | -ADHD | 0.0309* |
| 6 | 79 | IFJp | Area_IFJp | Fr | ContB | 0.000034 | +0.000034 | +ADHD | 0.0006* |
| 7 | 143 | IP2 | Area_IntraParietal_2 | Par | ContB | 0.000034 | -0.000034 | -ADHD | 0.0002* |
| 8 | 84 | a9-46v | Area_anterior_9-46v | Fr | ContC | 0.000033 | -0.000033 | -ADHD | 0.0006* |
| 9 | 49 | MIP | Medial_IntraParietal_Area | Par | DorsAttnA | 0.000030 | +0.000030 | +ADHD | 0.0004* |
| 10 | 116 | AIP | Anterior_IntraParietal_Area | Par | ContB | 0.000030 | -0.000029 | -ADHD | 0.2634 |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 72 | 8C (Area_8C) | ContC | +0.000044 | 0.0057* |
| 144 | IP1 (Area_IntraParietal_1) | ContB | +0.000036 | 0.0021* |
| 79 | IFJp (Area_IFJp) | ContB | +0.000034 | 0.0006* |
| 49 | MIP (Medial_IntraParietal_Area) | DorsAttnA | +0.000030 | 0.0004* |
| 169 | p10p (Area_posterior_10p) | ContC | +0.000023 | 0.0048* |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 170 | p47r (Area_posterior_47r) | ContC | -0.000060 | 0.0014* |
| 136 | PHT (Area_PHT) | ContB | -0.000035 | 0.0006* |
| 115 | PFt (Area_PFt) | DorsAttnB | -0.000035 | 0.0309* |
| 143 | IP2 (Area_IntraParietal_2) | ContB | -0.000034 | 0.0002* |
| 84 | a9-46v (Area_anterior_9-46v) | ContC | -0.000033 | 0.0006* |

---

## Expert: Salience (44 ROIs)

Networks: SalVentAttnA, SalVentAttnB, Limbic_TempPole, Limbic_OFC

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the Salience expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **Limbic_OFC** | 8 | 0.000026 | -0.000003 | -ADHD | p=0.0047 ** |
| **SalVentAttnA** | 19 | 0.000018 | -0.000006 | -ADHD | p=0.0045 ** |
| **Limbic_TempPole** | 9 | 0.000017 | +0.000002 | +ADHD | p=0.1055 n.s. |
| **SalVentAttnB** | 8 | 0.000015 | -0.000005 | -ADHD | p=0.0029 ** |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 90 | 11l | Area_11l | Fr | Limbic_OFC | 0.000060 | -0.000060 | -ADHD | 0.0058* |
| 2 | 108 | MI | Middle_Insular_Area | Ins | SalVentAttnA | 0.000041 | -0.000041 | -ADHD | 0.0014* |
| 3 | 98 | 43 | Area_43 | Fr | SalVentAttnA | 0.000040 | -0.000040 | -ADHD | 0.0010* |
| 4 | 163 | 25 | Area_25 | Fr | Limbic_OFC | 0.000036 | +0.000036 | +ADHD | 0.0009* |
| 5 | 89 | 10pp | Polar_10p | Fr | Limbic_OFC | 0.000030 | +0.000030 | +ADHD | 0.0113* |
| 6 | 133 | TE2a | Area_TE2_anterior | Temp | Limbic_TempPole | 0.000028 | -0.000028 | -ADHD | 0.0002* |
| 7 | 57 | 33pr | Area_33_prime | Fr | SalVentAttnA | 0.000028 | -0.000028 | -ADHD | 0.0020* |
| 8 | 171 | TGv | Area_TG_Ventral | Temp | Limbic_TempPole | 0.000026 | +0.000026 | +ADHD | 0.0212* |
| 9 | 36 | 5mv | Area_5m_ventral | Par | SalVentAttnA | 0.000026 | -0.000026 | -ADHD | 0.0019* |
| 10 | 166 | PoI1 | Area_Posterior_Insular_1 | Ins | SalVentAttnA | 0.000025 | +0.000025 | +ADHD | 0.0138* |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 163 | 25 (Area_25) | Limbic_OFC | +0.000036 | 0.0009* |
| 89 | 10pp (Polar_10p) | Limbic_OFC | +0.000030 | 0.0113* |
| 171 | TGv (Area_TG_Ventral) | Limbic_TempPole | +0.000026 | 0.0212* |
| 166 | PoI1 (Area_Posterior_Insular_1) | SalVentAttnA | +0.000025 | 0.0138* |
| 91 | 13l (Area_13l) | Limbic_OFC | +0.000023 | 0.0007* |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 90 | 11l (Area_11l) | Limbic_OFC | -0.000060 | 0.0058* |
| 108 | MI (Middle_Insular_Area) | SalVentAttnA | -0.000041 | 0.0014* |
| 98 | 43 (Area_43) | SalVentAttnA | -0.000040 | 0.0010* |
| 133 | TE2a (Area_TE2_anterior) | Limbic_TempPole | -0.000028 | 0.0002* |
| 57 | 33pr (Area_33_prime) | SalVentAttnA | -0.000028 | 0.0020* |

---

## Expert: SensoriMotor (43 ROIs)

Networks: VisCent, VisPeri, SomMotA, SomMotB

### Intra-Circuit Network Saliency

Which Yeo-17 sub-networks does the SensoriMotor expert rely on most for its representation?

| Network | n_ROIs | Abs Saliency | Signed | Direction | ADHD+ vs ADHD- (signed p) |
|---------|:------:|:------------:|:------:|:---------:|:------------------------:|
| **VisPeri** | 8 | 0.000029 | -0.000017 | -ADHD | p=0.0566 n.s. |
| **SomMotA** | 9 | 0.000028 | -0.000022 | -ADHD | p=0.0037 ** |
| **VisCent** | 12 | 0.000026 | -0.000015 | -ADHD | p=0.0198 * |
| **SomMotB** | 14 | 0.000023 | +0.000006 | +ADHD | p=0.9092 n.s. |

### Top 10 ROIs by Absolute Saliency

| Rank | ROI | Region | Long Name | Lobe | Network | Abs Saliency | Signed | Direction | Signed p |
|:----:|:---:|--------|-----------|------|---------|:----------:|:------:|:---------:|:--------:|
| 1 | 5 | V4 | Fourth_Visual_Area | Occ | VisCent | 0.000059 | -0.000059 | -ADHD | 0.0099* |
| 2 | 167 | Ig | Insular_Granular_Complex | Ins | SomMotB | 0.000059 | +0.000059 | +ADHD | 0.0072* |
| 3 | 6 | V8 | Eighth_Visual_Area | Occ | VisCent | 0.000051 | -0.000051 | -ADHD | 0.0035* |
| 4 | 152 | VMV1 | VentroMedial_Visual_Area_1 | Occ | VisPeri | 0.000051 | -0.000051 | -ADHD | 0.0140* |
| 5 | 35 | 5m | Area_5m | Par | SomMotA | 0.000049 | -0.000049 | -ADHD | 0.0049* |
| 6 | 3 | V2 | Second_Visual_Area | Occ | VisPeri | 0.000047 | -0.000047 | -ADHD | 0.0135* |
| 7 | 51 | 2 | Area_2 | Par | SomMotA | 0.000044 | -0.000044 | -ADHD | 0.9421 |
| 8 | 123 | PBelt | ParaBelt_Complex | Temp | SomMotB | 0.000044 | +0.000044 | +ADHD | 0.0140* |
| 9 | 15 | V7 | Seventh_Visual_Area | Occ | VisCent | 0.000043 | -0.000043 | -ADHD | 0.1935 |
| 10 | 23 | A1 | Primary_Auditory_Cortex | Temp | SomMotB | 0.000040 | -0.000040 | -ADHD | 0.0027* |

### Top 5 +ADHD ROIs (strongest positive association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 167 | Ig (Insular_Granular_Complex) | SomMotB | +0.000059 | 0.0072* |
| 123 | PBelt (ParaBelt_Complex) | SomMotB | +0.000044 | 0.0140* |
| 174 | A4 (Auditory_4_Complex) | SomMotB | +0.000035 | 0.0481* |
| 153 | VMV3 (VentroMedial_Visual_Area_3) | VisCent | +0.000029 | 0.0015* |
| 102 | 52 (Area_52) | SomMotB | +0.000024 | 0.1169 |

### Top 5 -ADHD ROIs (strongest negative association)

| ROI | Region | Network | Signed Saliency | p-value |
|:---:|--------|---------|:---------------:|:-------:|
| 5 | V4 (Fourth_Visual_Area) | VisCent | -0.000059 | 0.0099* |
| 6 | V8 (Eighth_Visual_Area) | VisCent | -0.000051 | 0.0035* |
| 152 | VMV1 (VentroMedial_Visual_Area_1) | VisPeri | -0.000051 | 0.0140* |
| 35 | 5m (Area_5m) | SomMotA | -0.000049 | 0.0049* |
| 3 | V2 (Second_Visual_Area) | VisPeri | -0.000047 | 0.0135* |

---

## Cross-Expert Comparison

### Dominant Network Per Expert

Which sub-network does each expert prioritize most?

| Expert | Top Network (abs) | Abs Saliency | Top Network (signed) | Signed | Direction |
|--------|:-----------------:|:------------:|:--------------------:|:------:|:---------:|
| DMN | DefaultA | 0.000032 | DefaultB | +0.000020 | +ADHD |
| Executive | ContC | 0.000024 | DorsAttnB | -0.000007 | -ADHD |
| Salience | Limbic_OFC | 0.000026 | SalVentAttnA | -0.000006 | -ADHD |
| SensoriMotor | VisPeri | 0.000029 | SomMotA | -0.000022 | -ADHD |

### Intra-Circuit Feature Hierarchy (All Experts)

Networks ranked by absolute saliency within each expert. This reveals whether experts prioritize their 'expected' sub-networks or learn unexpected feature hierarchies.

- **DMN**: DefaultA (0.0000, -ADHD) > DefaultB (0.0000, +ADHD) > DefaultC (0.0000, +ADHD) > TempPar (0.0000, -ADHD)
- **Executive**: ContC (0.0000, -ADHD) > ContB (0.0000, -ADHD) > DorsAttnB (0.0000, -ADHD) > DorsAttnA (0.0000, +ADHD) > ContA (0.0000, -ADHD)
- **Salience**: Limbic_OFC (0.0000, -ADHD) > SalVentAttnA (0.0000, -ADHD) > Limbic_TempPole (0.0000, +ADHD) > SalVentAttnB (0.0000, -ADHD)
- **SensoriMotor**: VisPeri (0.0000, -ADHD) > SomMotA (0.0000, -ADHD) > VisCent (0.0000, -ADHD) > SomMotB (0.0000, +ADHD)

---

## Significant Findings (p < 0.05)

- **DMN expert → DefaultB**: Signed saliency differs between ADHD+ and ADHD- (p=0.0117, direction=+ADHD)
- **DMN expert → DefaultC**: Signed saliency differs between ADHD+ and ADHD- (p=0.0020, direction=+ADHD)
- **DMN expert → TempPar**: Signed saliency differs between ADHD+ and ADHD- (p=0.0104, direction=-ADHD)
- **DMN expert → ROI PHA2** (DefaultB): p=0.0004, +ADHD, signed=+0.000049
- **DMN expert → ROI STSdp** (DefaultA): p=0.0008, -ADHD, signed=-0.000055
- **DMN expert → ROI TE1a** (TempPar): p=0.0009, -ADHD, signed=-0.000025
- **Executive expert → ContB**: Signed saliency differs between ADHD+ and ADHD- (p=0.0298, direction=-ADHD)
- **Executive expert → ContC**: Signed saliency differs between ADHD+ and ADHD- (p=0.0007, direction=-ADHD)
- **Executive expert → DorsAttnA**: Signed saliency differs between ADHD+ and ADHD- (p=0.0419, direction=+ADHD)
- **Executive expert → ROI 7AL** (DorsAttnB): p=0.0002, +ADHD, signed=+0.000007
- **Executive expert → ROI 8Av** (ContC): p=0.0002, +ADHD, signed=+0.000004
- **Executive expert → ROI PFm** (ContC): p=0.0002, -ADHD, signed=-0.000021
- **Salience expert → SalVentAttnA**: Signed saliency differs between ADHD+ and ADHD- (p=0.0045, direction=-ADHD)
- **Salience expert → SalVentAttnB**: Signed saliency differs between ADHD+ and ADHD- (p=0.0029, direction=-ADHD)
- **Salience expert → Limbic_OFC**: Signed saliency differs between ADHD+ and ADHD- (p=0.0047, direction=-ADHD)
- **Salience expert → ROI 23c** (SalVentAttnA): p=0.0001, +ADHD, signed=+0.000022
- **Salience expert → ROI 10v** (Limbic_OFC): p=0.0001, -ADHD, signed=-0.000019
- **Salience expert → ROI TE2a** (Limbic_TempPole): p=0.0002, -ADHD, signed=-0.000028
- **SensoriMotor expert → VisCent**: Signed saliency differs between ADHD+ and ADHD- (p=0.0198, direction=-ADHD)
- **SensoriMotor expert → SomMotA**: Signed saliency differs between ADHD+ and ADHD- (p=0.0037, direction=-ADHD)
- **SensoriMotor expert → ROI VMV2** (VisPeri): p=0.0009, +ADHD, signed=+0.000018
- **SensoriMotor expert → ROI LO2** (VisCent): p=0.0010, +ADHD, signed=+0.000019
- **SensoriMotor expert → ROI VMV3** (VisCent): p=0.0015, +ADHD, signed=+0.000029
