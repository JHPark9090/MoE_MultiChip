# Circuit MoE Interpretability Analysis

**Model**: quantum | **Config**: adhd_3 | **Experts**: 4

## 1. Circuit-Level Analysis: Gate Weights by Class

Gate weights indicate how much each circuit expert contributes to the final prediction. Differences between ADHD+ and ADHD- subjects reveal which circuits are differentially engaged.

| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |
|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|
| DMN | 0.2486 | 0.2488 | -0.0002 | -1.573 | 0.1163 | No significant difference |
| Executive | 0.2511 | 0.2514 | -0.0003 | -1.416 | 0.1575 | No significant difference |
| Salience | 0.2482 | 0.2477 | +0.0004 | 1.687 | 0.0922 | No significant difference |
| SensoriMotor | 0.2521 | 0.2521 | +0.0000 | 0.680 | 0.4971 | No significant difference |

## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)

Absolute gradient saliency measures how much each ROI's input signal influences the model output (magnitude only). Higher = more influential.

| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| DMN | 0.000841 | 0.000823 | 0.000854 | -0.000031 | 41 |
| Executive | 0.000566 | 0.000558 | 0.000573 | -0.000014 | 52 |
| Salience | 0.000786 | 0.000774 | 0.000795 | -0.000021 | 44 |
| SensoriMotor | 0.000819 | 0.000802 | 0.000831 | -0.000029 | 43 |

## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)

Signed gradient saliency preserves the direction of influence. **Positive** = increasing this circuit's signal pushes the prediction toward ADHD+. **Negative** = pushes toward ADHD-.

| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| DMN | +0.000067 | +0.000063 | +0.000070 | positive (+ADHD) | 41 |
| Executive | -0.000026 | -0.000028 | -0.000026 | negative (-ADHD) | 52 |
| Salience | -0.000001 | +0.000004 | -0.000005 | negative (-ADHD) | 44 |
| SensoriMotor | +0.000110 | +0.000100 | +0.000118 | positive (+ADHD) | 43 |

## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network

Absolute saliency aggregated by Yeo-17 network (magnitude of influence).

| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| DefaultC | 0.000873 | 0.000854 | 0.000887 | -0.000034 | 16 |
| SomMotB | 0.000845 | 0.000828 | 0.000859 | -0.000031 | 14 |
| SomMotA | 0.000838 | 0.000824 | 0.000848 | -0.000024 | 9 |
| Limbic_TempPole | 0.000836 | 0.000821 | 0.000848 | -0.000027 | 9 |
| DefaultB | 0.000835 | 0.000815 | 0.000850 | -0.000035 | 5 |
| Limbic_OFC | 0.000825 | 0.000810 | 0.000837 | -0.000027 | 8 |
| TempPar | 0.000822 | 0.000807 | 0.000834 | -0.000027 | 15 |
| VisCent | 0.000800 | 0.000781 | 0.000815 | -0.000034 | 12 |
| DefaultA | 0.000800 | 0.000781 | 0.000814 | -0.000034 | 5 |
| VisPeri | 0.000778 | 0.000764 | 0.000789 | -0.000025 | 8 |
| SalVentAttnA | 0.000758 | 0.000748 | 0.000766 | -0.000018 | 19 |
| SalVentAttnB | 0.000758 | 0.000749 | 0.000765 | -0.000016 | 8 |
| ContB | 0.000593 | 0.000584 | 0.000600 | -0.000016 | 12 |
| ContC | 0.000569 | 0.000560 | 0.000575 | -0.000016 | 11 |
| DorsAttnA | 0.000564 | 0.000556 | 0.000570 | -0.000014 | 16 |
| DorsAttnB | 0.000555 | 0.000546 | 0.000561 | -0.000015 | 8 |
| ContA | 0.000526 | 0.000521 | 0.000530 | -0.000009 | 5 |

## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency by Yeo-17 network. Positive = network activity positively associated with ADHD prediction. Negative = negatively associated.

| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| VisCent | +0.000141 | +0.000131 | +0.000150 | +ADHD | 12 |
| DefaultC | +0.000131 | +0.000126 | +0.000135 | +ADHD | 16 |
| SomMotB | +0.000131 | +0.000117 | +0.000141 | +ADHD | 14 |
| SalVentAttnB | +0.000081 | +0.000084 | +0.000079 | +ADHD | 8 |
| SomMotA | +0.000077 | +0.000073 | +0.000080 | +ADHD | 9 |
| VisPeri | +0.000065 | +0.000057 | +0.000072 | +ADHD | 8 |
| DefaultA | +0.000055 | +0.000048 | +0.000060 | +ADHD | 5 |
| SalVentAttnA | +0.000043 | +0.000046 | +0.000041 | +ADHD | 19 |
| DefaultB | +0.000030 | +0.000024 | +0.000035 | +ADHD | 5 |
| ContA | +0.000015 | +0.000013 | +0.000016 | +ADHD | 5 |
| TempPar | +0.000014 | +0.000013 | +0.000015 | +ADHD | 15 |
| ContC | +0.000011 | +0.000011 | +0.000011 | +ADHD | 11 |
| DorsAttnB | -0.000021 | -0.000028 | -0.000016 | -ADHD | 8 |
| Limbic_TempPole | -0.000043 | -0.000045 | -0.000042 | -ADHD | 9 |
| DorsAttnA | -0.000044 | -0.000045 | -0.000044 | -ADHD | 16 |
| ContB | -0.000057 | -0.000056 | -0.000058 | -ADHD | 12 |
| Limbic_OFC | -0.000142 | -0.000123 | -0.000156 | -ADHD | 8 |

## 4. Network-Level Analysis: Input Projection Weights per Expert

Weight magnitudes of the first linear layer in each expert, grouped by Yeo-17 network. Shows which sub-networks each expert learns to attend to.

### Expert: DMN

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| DefaultC | 0.7482 | 0.0411 | 0.8160 | 16 |
| DefaultB | 0.7302 | 0.0645 | 0.8184 | 5 |
| TempPar | 0.7261 | 0.0664 | 0.8852 | 15 |
| DefaultA | 0.7248 | 0.0612 | 0.8090 | 5 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 70 | 9p | TempPar | 0.8852 |
| 126 | PHA3 | DefaultB | 0.8184 |
| 149 | PGi | DefaultC | 0.8160 |
| 32 | v23ab | DefaultC | 0.8102 |
| 128 | STSdp | DefaultA | 0.8090 |

### Expert: Executive

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| ContC | 0.7160 | 0.0396 | 0.7932 | 11 |
| ContB | 0.7160 | 0.0521 | 0.8293 | 12 |
| DorsAttnB | 0.6948 | 0.0481 | 0.7647 | 8 |
| DorsAttnA | 0.6843 | 0.0419 | 0.7426 | 16 |
| ContA | 0.6752 | 0.0360 | 0.7311 | 5 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 80 | IFSp | ContB | 0.8293 |
| 148 | PFm | ContC | 0.7932 |
| 88 | a10p | ContC | 0.7855 |
| 77 | 6r | ContB | 0.7853 |
| 38 | 5L | DorsAttnB | 0.7647 |

### Expert: Salience

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| Limbic_OFC | 0.7309 | 0.0665 | 0.8306 | 8 |
| SalVentAttnB | 0.7177 | 0.0256 | 0.7509 | 8 |
| Limbic_TempPole | 0.7171 | 0.0869 | 0.9030 | 9 |
| SalVentAttnA | 0.6974 | 0.0489 | 0.7774 | 19 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 134 | TF | Limbic_TempPole | 0.9030 |
| 92 | OFC | Limbic_OFC | 0.8306 |
| 171 | TGv | Limbic_TempPole | 0.8063 |
| 164 | s32 | Limbic_OFC | 0.7890 |
| 165 | pOFC | Limbic_OFC | 0.7776 |

### Expert: SensoriMotor

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| SomMotB | 0.7205 | 0.0519 | 0.8222 | 14 |
| SomMotA | 0.7194 | 0.0480 | 0.8048 | 9 |
| VisCent | 0.6900 | 0.0520 | 0.7707 | 12 |
| VisPeri | 0.6815 | 0.0522 | 0.7626 | 8 |

**Top 5 ROIs by weight magnitude:**

| ROI | Region | Network | Weight Norm |
|:---:|--------|---------|:----------:|
| 23 | A1 | SomMotB | 0.8222 |
| 39 | 24dd | SomMotA | 0.8048 |
| 53 | 6d | SomMotA | 0.7818 |
| 174 | A4 | SomMotB | 0.7803 |
| 4 | V3 | VisCent | 0.7707 |

## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency

| Rank | ROI | Region | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |
|:----:|:---:|--------|---------|---------|:--------:|:-----:|:-----:|:----:|
| 1 | 134 | TF | Limbic_TempPole | Salience | 0.001220 | 0.001194 | 0.001239 | -0.000044 |
| 2 | 175 | STSva | TempPar | DMN | 0.001153 | 0.001131 | 0.001169 | -0.000038 |
| 3 | 171 | TGv | Limbic_TempPole | Salience | 0.001134 | 0.001114 | 0.001150 | -0.000036 |
| 4 | 23 | A1 | SomMotB | SensoriMotor | 0.001093 | 0.001074 | 0.001108 | -0.000033 |
| 5 | 32 | v23ab | DefaultC | DMN | 0.001069 | 0.001035 | 0.001095 | -0.000059 |
| 6 | 92 | OFC | Limbic_OFC | Salience | 0.001067 | 0.001043 | 0.001086 | -0.000043 |
| 7 | 129 | STSvp | TempPar | DMN | 0.001036 | 0.000995 | 0.001067 | -0.000072 |
| 8 | 70 | 9p | TempPar | DMN | 0.001010 | 0.000998 | 0.001020 | -0.000022 |
| 9 | 21 | PIT | VisCent | SensoriMotor | 0.001001 | 0.000980 | 0.001017 | -0.000037 |
| 10 | 128 | STSdp | DefaultA | DMN | 0.000996 | 0.000965 | 0.001019 | -0.000054 |
| 11 | 67 | 8Ad | DefaultC | DMN | 0.000967 | 0.000945 | 0.000985 | -0.000040 |
| 12 | 106 | TA2 | SomMotB | SensoriMotor | 0.000959 | 0.000932 | 0.000979 | -0.000047 |
| 13 | 149 | PGi | DefaultC | DMN | 0.000951 | 0.000939 | 0.000960 | -0.000021 |
| 14 | 126 | PHA3 | DefaultB | DMN | 0.000946 | 0.000932 | 0.000957 | -0.000024 |
| 15 | 31 | 23d | DefaultC | DMN | 0.000940 | 0.000915 | 0.000959 | -0.000044 |
| 16 | 73 | 44 | TempPar | DMN | 0.000934 | 0.000911 | 0.000951 | -0.000040 |
| 17 | 39 | 24dd | SomMotA | SensoriMotor | 0.000929 | 0.000914 | 0.000941 | -0.000027 |
| 18 | 29 | 7m | DefaultC | DMN | 0.000928 | 0.000909 | 0.000942 | -0.000034 |
| 19 | 125 | PHA1 | DefaultB | DMN | 0.000927 | 0.000899 | 0.000948 | -0.000049 |
| 20 | 8 | 3b | SomMotA | SensoriMotor | 0.000923 | 0.000893 | 0.000945 | -0.000052 |

## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences

ROIs where absolute gradient saliency differs most between classes. Positive diff = more salient for ADHD+.

| Rank | ROI | Region | Network | Circuit | Diff | ADHD+ | ADHD- |
|:----:|:---:|--------|---------|---------|:----:|:-----:|:-----:|
| 1 | 129 | STSvp | TempPar | DMN | -0.000072 | 0.000995 | 0.001067 |
| 2 | 32 | v23ab | DefaultC | DMN | -0.000059 | 0.001035 | 0.001095 |
| 3 | 64 | 10r | DefaultC | DMN | -0.000055 | 0.000881 | 0.000936 |
| 4 | 128 | STSdp | DefaultA | DMN | -0.000054 | 0.000965 | 0.001019 |
| 5 | 8 | 3b | SomMotA | SensoriMotor | -0.000052 | 0.000893 | 0.000945 |
| 6 | 125 | PHA1 | DefaultB | DMN | -0.000049 | 0.000899 | 0.000948 |
| 7 | 35 | 5m | SomMotA | SensoriMotor | -0.000048 | 0.000841 | 0.000889 |
| 8 | 106 | TA2 | SomMotB | SensoriMotor | -0.000047 | 0.000932 | 0.000979 |
| 9 | 61 | d32 | DefaultC | DMN | -0.000047 | 0.000890 | 0.000937 |
| 10 | 138 | TPOJ1 | DefaultA | DMN | -0.000045 | 0.000741 | 0.000786 |
| 11 | 134 | TF | Limbic_TempPole | Salience | -0.000044 | 0.001194 | 0.001239 |
| 12 | 31 | 23d | DefaultC | DMN | -0.000044 | 0.000915 | 0.000959 |
| 13 | 102 | 52 | SomMotB | SensoriMotor | -0.000044 | 0.000761 | 0.000805 |
| 14 | 15 | V7 | VisCent | SensoriMotor | -0.000043 | 0.000851 | 0.000894 |
| 15 | 92 | OFC | Limbic_OFC | Salience | -0.000043 | 0.001043 | 0.001086 |
| 16 | 4 | V3 | VisCent | SensoriMotor | -0.000043 | 0.000885 | 0.000927 |
| 17 | 162 | VVC | VisCent | SensoriMotor | -0.000042 | 0.000803 | 0.000845 |
| 18 | 154 | PHA2 | DefaultB | DMN | -0.000042 | 0.000792 | 0.000834 |
| 19 | 152 | VMV1 | VisPeri | SensoriMotor | -0.000040 | 0.000754 | 0.000794 |
| 20 | 67 | 8Ad | DefaultC | DMN | -0.000040 | 0.000945 | 0.000985 |

## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency shows the direction of each ROI's relationship with ADHD. **Positive** = increasing this ROI's signal pushes prediction toward ADHD+. **Negative** = pushes toward ADHD-.

### Top 10 ROIs with Strongest Positive (+ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 21 | PIT | VisCent | SensoriMotor | +0.000531 | +0.000513 | +0.000545 |
| 2 | 175 | STSva | TempPar | DMN | +0.000496 | +0.000468 | +0.000518 |
| 3 | 23 | A1 | SomMotB | SensoriMotor | +0.000495 | +0.000493 | +0.000495 |
| 4 | 135 | TE2p | Limbic_TempPole | Salience | +0.000472 | +0.000449 | +0.000489 |
| 5 | 110 | AVI | SalVentAttnB | Salience | +0.000417 | +0.000399 | +0.000431 |
| 6 | 128 | STSdp | DefaultA | DMN | +0.000414 | +0.000375 | +0.000444 |
| 7 | 37 | 23c | SalVentAttnA | Salience | +0.000413 | +0.000404 | +0.000419 |
| 8 | 106 | TA2 | SomMotB | SensoriMotor | +0.000384 | +0.000357 | +0.000404 |
| 9 | 101 | OP2-3 | SomMotB | SensoriMotor | +0.000380 | +0.000379 | +0.000382 |
| 10 | 64 | 10r | DefaultC | DMN | +0.000376 | +0.000347 | +0.000398 |

### Top 10 ROIs with Strongest Negative (-ADHD) Relationship

| Rank | ROI | Region | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|--------|---------|---------|:------------:|:------------:|:------------:|
| 1 | 91 | 13l | Limbic_OFC | Salience | -0.000433 | -0.000393 | -0.000463 |
| 2 | 59 | p32pr | SalVentAttnA | Salience | -0.000370 | -0.000343 | -0.000390 |
| 3 | 147 | PF | SalVentAttnA | Salience | -0.000327 | -0.000305 | -0.000344 |
| 4 | 65 | 47m | TempPar | DMN | -0.000319 | -0.000314 | -0.000322 |
| 5 | 164 | s32 | Limbic_OFC | Salience | -0.000311 | -0.000265 | -0.000346 |
| 6 | 89 | 10pp | Limbic_OFC | Salience | -0.000310 | -0.000291 | -0.000324 |
| 7 | 8 | 3b | SomMotA | SensoriMotor | -0.000307 | -0.000298 | -0.000313 |
| 8 | 34 | 31pv | DefaultC | DMN | -0.000266 | -0.000275 | -0.000259 |
| 9 | 117 | EC | Limbic_TempPole | Salience | -0.000249 | -0.000219 | -0.000271 |
| 10 | 121 | PeEc | Limbic_TempPole | Salience | -0.000245 | -0.000228 | -0.000259 |
