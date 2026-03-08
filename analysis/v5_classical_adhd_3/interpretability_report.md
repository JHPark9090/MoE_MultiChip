# Circuit MoE Interpretability Analysis

**Model**: classical | **Config**: adhd_3 | **Experts**: 4

## 1. Circuit-Level Analysis: Gate Weights by Class

Gate weights indicate how much each circuit expert contributes to the final prediction. Differences between ADHD+ and ADHD- subjects reveal which circuits are differentially engaged.

| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |
|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|
| DMN | 0.2497 | 0.2496 | +0.0002 | 1.822 | 0.0690 | No significant difference |
| Executive | 0.2517 | 0.2517 | -0.0000 | -0.785 | 0.4326 | No significant difference |
| Salience | 0.2479 | 0.2479 | +0.0000 | 0.752 | 0.4526 | No significant difference |
| SensoriMotor | 0.2506 | 0.2508 | -0.0002 | -1.629 | 0.1038 | No significant difference |

## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)

Absolute gradient saliency measures how much each ROI's input signal influences the model output (magnitude only). Higher = more influential.

| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| DMN | 0.000012 | 0.000013 | 0.000012 | +0.000000 | 41 |
| Executive | 0.000013 | 0.000013 | 0.000013 | +0.000000 | 52 |
| Salience | 0.000012 | 0.000013 | 0.000012 | +0.000000 | 44 |
| SensoriMotor | 0.000014 | 0.000014 | 0.000014 | +0.000000 | 43 |

## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)

Signed gradient saliency preserves the direction of influence. **Positive** = increasing this circuit's signal pushes the prediction toward ADHD+. **Negative** = pushes toward ADHD-.

| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| DMN | +0.000001 | +0.000001 | +0.000001 | positive (+ADHD) | 41 |
| Executive | +0.000005 | +0.000005 | +0.000005 | positive (+ADHD) | 52 |
| Salience | +0.000001 | +0.000001 | +0.000002 | positive (+ADHD) | 44 |
| SensoriMotor | +0.000000 | +0.000000 | +0.000000 | positive (+ADHD) | 43 |

## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network

Absolute saliency aggregated by Yeo-17 network (magnitude of influence).

| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| ContA | 0.000023 | 0.000023 | 0.000022 | +0.000001 | 5 |
| DefaultA | 0.000017 | 0.000017 | 0.000017 | +0.000000 | 5 |
| VisCent | 0.000017 | 0.000017 | 0.000017 | +0.000000 | 12 |
| SomMotA | 0.000015 | 0.000015 | 0.000015 | +0.000000 | 9 |
| Limbic_OFC | 0.000015 | 0.000015 | 0.000014 | +0.000000 | 8 |
| ContB | 0.000014 | 0.000014 | 0.000014 | +0.000000 | 12 |
| SalVentAttnB | 0.000014 | 0.000014 | 0.000014 | +0.000000 | 8 |
| Limbic_TempPole | 0.000013 | 0.000013 | 0.000013 | +0.000000 | 9 |
| DorsAttnB | 0.000012 | 0.000013 | 0.000012 | +0.000000 | 8 |
| DefaultC | 0.000012 | 0.000012 | 0.000012 | +0.000000 | 16 |
| VisPeri | 0.000012 | 0.000012 | 0.000012 | +0.000000 | 8 |
| SomMotB | 0.000012 | 0.000012 | 0.000012 | +0.000000 | 14 |
| TempPar | 0.000012 | 0.000012 | 0.000011 | +0.000001 | 15 |
| DorsAttnA | 0.000012 | 0.000012 | 0.000012 | +0.000000 | 16 |
| DefaultB | 0.000010 | 0.000011 | 0.000010 | +0.000001 | 5 |
| SalVentAttnA | 0.000010 | 0.000011 | 0.000010 | +0.000001 | 19 |
| ContC | 0.000008 | 0.000009 | 0.000008 | +0.000000 | 11 |

## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency by Yeo-17 network. Positive = network activity positively associated with ADHD prediction. Negative = negatively associated.

| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| ContA | +0.000022 | +0.000022 | +0.000021 | +ADHD | 5 |
| DefaultA | +0.000015 | +0.000015 | +0.000015 | +ADHD | 5 |
| DorsAttnB | +0.000009 | +0.000009 | +0.000009 | +ADHD | 8 |
| SalVentAttnB | +0.000007 | +0.000007 | +0.000007 | +ADHD | 8 |
| DorsAttnA | +0.000006 | +0.000006 | +0.000006 | +ADHD | 16 |
| SomMotB | +0.000005 | +0.000005 | +0.000005 | +ADHD | 14 |
| SalVentAttnA | +0.000002 | +0.000002 | +0.000002 | +ADHD | 19 |
| VisCent | +0.000002 | +0.000002 | +0.000002 | +ADHD | 12 |
| ContB | +0.000002 | +0.000002 | +0.000002 | +ADHD | 12 |
| DefaultB | +0.000001 | +0.000001 | +0.000001 | +ADHD | 5 |
| Limbic_TempPole | +0.000001 | +0.000001 | +0.000001 | +ADHD | 9 |
| DefaultC | -0.000000 | +0.000000 | -0.000000 | -ADHD | 16 |
| VisPeri | -0.000001 | -0.000001 | -0.000002 | -ADHD | 8 |
| TempPar | -0.000002 | -0.000002 | -0.000002 | -ADHD | 15 |
| ContC | -0.000003 | -0.000003 | -0.000003 | -ADHD | 11 |
| Limbic_OFC | -0.000005 | -0.000005 | -0.000005 | -ADHD | 8 |
| SomMotA | -0.000008 | -0.000008 | -0.000008 | -ADHD | 9 |

## 4. Network-Level Analysis: Input Projection Weights per Expert

Weight magnitudes of the first linear layer in each expert, grouped by Yeo-17 network. Shows which sub-networks each expert learns to attend to.

### Expert: DMN

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| DefaultA | 0.7275 | 0.0269 | 0.7791 | 5 |
| DefaultC | 0.7246 | 0.0322 | 0.8122 | 16 |
| TempPar | 0.7195 | 0.0407 | 0.7874 | 15 |
| DefaultB | 0.7116 | 0.0565 | 0.7940 | 5 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 32 | v23ab | DefaultC | 0.8122 |
| 126 | PHA3 | DefaultB | 0.7940 |
| 65 | 47m | TempPar | 0.7874 |
| 124 | A5 | DefaultA | 0.7791 |
| 175 | STSva | TempPar | 0.7698 |

### Expert: Executive

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| ContA | 0.6557 | 0.0235 | 0.6917 | 5 |
| DorsAttnA | 0.6529 | 0.0280 | 0.7108 | 16 |
| ContB | 0.6487 | 0.0324 | 0.6923 | 12 |
| DorsAttnB | 0.6481 | 0.0351 | 0.7111 | 8 |
| ContC | 0.6163 | 0.0351 | 0.6716 | 11 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 46 | 7PC | DorsAttnB | 0.7111 |
| 17 | FFC | DorsAttnA | 0.7108 |
| 143 | IP2 | ContB | 0.6923 |
| 14 | POS2 | ContA | 0.6917 |
| 48 | VIP | DorsAttnA | 0.6914 |

### Expert: Salience

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| Limbic_OFC | 0.7199 | 0.0187 | 0.7606 | 8 |
| Limbic_TempPole | 0.6971 | 0.0549 | 0.7951 | 9 |
| SalVentAttnB | 0.6950 | 0.0188 | 0.7172 | 8 |
| SalVentAttnA | 0.6852 | 0.0424 | 0.7630 | 19 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 134 | TF | Limbic_TempPole | 0.7951 |
| 166 | PoI1 | SalVentAttnA | 0.7630 |
| 89 | 10pp | Limbic_OFC | 0.7606 |
| 130 | TGd | Limbic_TempPole | 0.7553 |
| 98 | 43 | SalVentAttnA | 0.7431 |

### Expert: SensoriMotor

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| SomMotA | 0.7385 | 0.0485 | 0.8136 | 9 |
| VisCent | 0.7128 | 0.0348 | 0.7687 | 12 |
| SomMotB | 0.6990 | 0.0367 | 0.7716 | 14 |
| VisPeri | 0.6820 | 0.0330 | 0.7429 | 8 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 35 | 5m | SomMotA | 0.8136 |
| 50 | 1 | SomMotA | 0.7767 |
| 52 | 3a | SomMotA | 0.7757 |
| 103 | RI | SomMotB | 0.7716 |
| 15 | V7 | VisCent | 0.7687 |

## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency

| Rank | ROI | Region | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |
|:----:|:---:|--------|---------|---------|:--------:|:-----:|:-----:|:----:|
| 1 | 116 | AIP | ContB | Executive | 0.000038 | 0.000039 | 0.000038 | +0.000001 |
| 2 | 35 | 5m | SomMotA | SensoriMotor | 0.000036 | 0.000035 | 0.000036 | -0.000001 |
| 3 | 13 | RSC | ContA | Executive | 0.000034 | 0.000034 | 0.000034 | +0.000000 |
| 4 | 128 | STSdp | DefaultA | DMN | 0.000032 | 0.000033 | 0.000032 | +0.000000 |
| 5 | 41 | 7AL | DorsAttnB | Executive | 0.000032 | 0.000032 | 0.000031 | +0.000001 |
| 6 | 161 | 31a | ContA | Executive | 0.000030 | 0.000031 | 0.000030 | +0.000001 |
| 7 | 153 | VMV3 | VisCent | SensoriMotor | 0.000030 | 0.000030 | 0.000030 | -0.000000 |
| 8 | 64 | 10r | DefaultC | DMN | 0.000029 | 0.000029 | 0.000030 | -0.000000 |
| 9 | 20 | LO2 | VisCent | SensoriMotor | 0.000026 | 0.000026 | 0.000026 | +0.000000 |
| 10 | 158 | LO3 | DorsAttnA | Executive | 0.000025 | 0.000024 | 0.000025 | -0.000001 |
| 11 | 164 | s32 | Limbic_OFC | Salience | 0.000023 | 0.000023 | 0.000024 | -0.000001 |
| 12 | 178 | a32pr | SalVentAttnB | Salience | 0.000023 | 0.000023 | 0.000023 | -0.000001 |
| 13 | 17 | FFC | DorsAttnA | Executive | 0.000023 | 0.000023 | 0.000023 | +0.000000 |
| 14 | 123 | PBelt | SomMotB | SensoriMotor | 0.000022 | 0.000022 | 0.000023 | -0.000000 |
| 15 | 104 | PFcm | SalVentAttnA | Salience | 0.000022 | 0.000023 | 0.000022 | +0.000001 |
| 16 | 61 | d32 | DefaultC | DMN | 0.000022 | 0.000022 | 0.000022 | -0.000000 |
| 17 | 26 | PCV | ContA | Executive | 0.000021 | 0.000021 | 0.000021 | +0.000000 |
| 18 | 81 | IFSa | ContB | Executive | 0.000020 | 0.000020 | 0.000020 | -0.000000 |
| 19 | 14 | POS2 | ContA | Executive | 0.000020 | 0.000021 | 0.000020 | +0.000002 |
| 20 | 6 | V8 | VisCent | SensoriMotor | 0.000020 | 0.000020 | 0.000020 | -0.000001 |

## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences

ROIs where absolute gradient saliency differs most between classes. Positive diff = more salient for ADHD+.

| Rank | ROI | Region | Network | Circuit | Diff | ADHD+ | ADHD- |
|:----:|:---:|--------|---------|---------|:----:|:-----:|:-----:|
| 1 | 110 | AVI | SalVentAttnB | Salience | +0.000002 | 0.000011 | 0.000009 |
| 2 | 136 | PHT | ContB | Executive | +0.000002 | 0.000016 | 0.000015 |
| 3 | 14 | POS2 | ContA | Executive | +0.000002 | 0.000021 | 0.000020 |
| 4 | 146 | PFop | SalVentAttnA | Salience | +0.000002 | 0.000010 | 0.000009 |
| 5 | 134 | TF | Limbic_TempPole | Salience | +0.000002 | 0.000015 | 0.000013 |
| 6 | 132 | TE1p | ContC | Executive | +0.000001 | 0.000013 | 0.000011 |
| 7 | 82 | p9-46v | ContB | Executive | +0.000001 | 0.000009 | 0.000008 |
| 8 | 125 | PHA1 | DefaultB | DMN | +0.000001 | 0.000016 | 0.000015 |
| 9 | 97 | s6-8 | DefaultC | DMN | +0.000001 | 0.000012 | 0.000011 |
| 10 | 25 | SFL | TempPar | DMN | +0.000001 | 0.000009 | 0.000008 |
| 11 | 127 | STSda | TempPar | DMN | +0.000001 | 0.000018 | 0.000016 |
| 12 | 121 | PeEc | Limbic_TempPole | Salience | +0.000001 | 0.000009 | 0.000008 |
| 13 | 66 | 8Av | ContC | Executive | +0.000001 | 0.000010 | 0.000009 |
| 14 | 131 | TE1a | TempPar | DMN | +0.000001 | 0.000010 | 0.000009 |
| 15 | 93 | 47s | TempPar | DMN | +0.000001 | 0.000016 | 0.000015 |
| 16 | 158 | LO3 | DorsAttnA | Executive | -0.000001 | 0.000024 | 0.000025 |
| 17 | 95 | 6a | DorsAttnB | Executive | +0.000001 | 0.000009 | 0.000008 |
| 18 | 92 | OFC | Limbic_OFC | Salience | +0.000001 | 0.000010 | 0.000009 |
| 19 | 73 | 44 | TempPar | DMN | +0.000001 | 0.000019 | 0.000018 |
| 20 | 116 | AIP | ContB | Executive | +0.000001 | 0.000039 | 0.000038 |

## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency shows the direction of each ROI's relationship with ADHD. **Positive** = increasing this ROI's signal pushes prediction toward ADHD+. **Negative** = pushes toward ADHD-.

### Top 10 ROIs with Strongest Positive (+ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 116 | AIP | ContB | Executive | +0.000038 | +0.000039 | +0.000038 |
| 2 | 13 | RSC | ContA | Executive | +0.000034 | +0.000034 | +0.000034 |
| 3 | 128 | STSdp | DefaultA | DMN | +0.000032 | +0.000033 | +0.000032 |
| 4 | 41 | 7AL | DorsAttnB | Executive | +0.000031 | +0.000032 | +0.000031 |
| 5 | 153 | VMV3 | VisCent | SensoriMotor | +0.000030 | +0.000030 | +0.000030 |
| 6 | 161 | 31a | ContA | Executive | +0.000029 | +0.000029 | +0.000028 |
| 7 | 20 | LO2 | VisCent | SensoriMotor | +0.000026 | +0.000026 | +0.000026 |
| 8 | 158 | LO3 | DorsAttnA | Executive | +0.000025 | +0.000024 | +0.000025 |
| 9 | 178 | a32pr | SalVentAttnB | Salience | +0.000022 | +0.000022 | +0.000023 |
| 10 | 123 | PBelt | SomMotB | SensoriMotor | +0.000022 | +0.000022 | +0.000023 |

### Top 10 ROIs with Strongest Negative (-ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 35 | 5m | SomMotA | SensoriMotor | -0.000035 | -0.000034 | -0.000036 |
| 2 | 64 | 10r | DefaultC | DMN | -0.000029 | -0.000029 | -0.000030 |
| 3 | 164 | s32 | Limbic_OFC | Salience | -0.000023 | -0.000022 | -0.000023 |
| 4 | 81 | IFSa | ContB | Executive | -0.000020 | -0.000020 | -0.000020 |
| 5 | 6 | V8 | VisCent | SensoriMotor | -0.000020 | -0.000019 | -0.000020 |
| 6 | 54 | 6mp | SomMotA | SensoriMotor | -0.000020 | -0.000020 | -0.000020 |
| 7 | 73 | 44 | TempPar | DMN | -0.000019 | -0.000019 | -0.000018 |
| 8 | 18 | V3B | VisCent | SensoriMotor | -0.000019 | -0.000019 | -0.000019 |
| 9 | 51 | 2 | SomMotA | SensoriMotor | -0.000017 | -0.000018 | -0.000017 |
| 10 | 168 | FOP5 | SalVentAttnB | Salience | -0.000017 | -0.000017 | -0.000018 |
