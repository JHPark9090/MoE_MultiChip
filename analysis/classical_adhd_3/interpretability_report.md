# Circuit MoE Interpretability Analysis

> **DEPRECATED (2026-03-07):** Generated with incorrect Yeo-17 mapping. ROI/network/circuit interpretations are not neurobiologically valid. Will be regenerated with corrected mapping (v5).

**Model**: classical | **Config**: adhd_3 | **Experts**: 4

## 1. Circuit-Level Analysis: Gate Weights by Class

Gate weights indicate how much each circuit expert contributes to the final prediction. Differences between ADHD+ and ADHD- subjects reveal which circuits are differentially engaged.

| Circuit | ADHD+ Mean | ADHD- Mean | Diff (pos-neg) | t-stat | p-value | Interpretation |
|---------|:----------:|:----------:|:--------------:|:------:|:-------:|----------------|
| DMN | 0.2505 | 0.2502 | +0.0002 | 0.370 | 0.7116 | No significant difference |
| Executive | 0.2487 | 0.2496 | -0.0009 | -1.082 | 0.2797 | No significant difference |
| Salience | 0.2587 | 0.2578 | +0.0010 | 0.612 | 0.5407 | No significant difference |
| SensoriMotor | 0.2421 | 0.2424 | -0.0004 | -0.754 | 0.4512 | No significant difference |

## 2. Circuit-Level Analysis: Gradient Saliency (Absolute)

Absolute gradient saliency measures how much each ROI's input signal influences the model output (magnitude only). Higher = more influential.

| Circuit | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| DMN | 0.000387 | 0.000380 | 0.000393 | -0.000014 | 55 |
| Executive | 0.000183 | 0.000183 | 0.000184 | -0.000000 | 50 |
| Salience | 0.000443 | 0.000451 | 0.000437 | +0.000014 | 29 |
| SensoriMotor | 0.000515 | 0.000497 | 0.000529 | -0.000033 | 46 |

## 2b. Circuit-Level Analysis: Signed Gradient Saliency (Direction)

Signed gradient saliency preserves the direction of influence. **Positive** = increasing this circuit's signal pushes the prediction toward ADHD+. **Negative** = pushes toward ADHD-.

| Circuit | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| DMN | +0.000040 | +0.000036 | +0.000044 | positive (+ADHD) | 55 |
| Executive | +0.000017 | +0.000017 | +0.000017 | positive (+ADHD) | 50 |
| Salience | +0.000009 | +0.000008 | +0.000009 | positive (+ADHD) | 29 |
| SensoriMotor | +0.000115 | +0.000115 | +0.000115 | positive (+ADHD) | 46 |

## 3. Network-Level Analysis: Gradient Saliency by Yeo-17 Network

Absolute saliency aggregated by Yeo-17 network (magnitude of influence).

| Network | Saliency (all) | ADHD+ | ADHD- | Diff | n_ROIs |
|---------|:--------------:|:-----:|:-----:|:----:|:------:|
| SalVentAttnB | 0.000623 | 0.000635 | 0.000614 | +0.000020 | 6 |
| VisPeri | 0.000554 | 0.000531 | 0.000571 | -0.000040 | 12 |
| SomMotA | 0.000522 | 0.000502 | 0.000537 | -0.000035 | 16 |
| SomMotB | 0.000517 | 0.000503 | 0.000528 | -0.000024 | 8 |
| Limbic_OFC | 0.000505 | 0.000517 | 0.000497 | +0.000020 | 4 |
| VisCent | 0.000457 | 0.000441 | 0.000468 | -0.000027 | 10 |
| Limbic_TempPole | 0.000430 | 0.000438 | 0.000424 | +0.000013 | 4 |
| DefaultC | 0.000416 | 0.000410 | 0.000421 | -0.000011 | 10 |
| DefaultA | 0.000399 | 0.000390 | 0.000406 | -0.000015 | 21 |
| DefaultB | 0.000388 | 0.000380 | 0.000395 | -0.000015 | 16 |
| SalVentAttnA | 0.000358 | 0.000364 | 0.000353 | +0.000011 | 15 |
| ContC | 0.000347 | 0.000335 | 0.000357 | -0.000022 | 4 |
| TempPar | 0.000318 | 0.000312 | 0.000322 | -0.000010 | 8 |
| ContA | 0.000217 | 0.000219 | 0.000216 | +0.000003 | 13 |
| DorsAttnA | 0.000163 | 0.000162 | 0.000163 | -0.000001 | 13 |
| ContB | 0.000155 | 0.000156 | 0.000155 | +0.000001 | 8 |
| DorsAttnB | 0.000133 | 0.000135 | 0.000132 | +0.000003 | 12 |

## 3b. Network-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency by Yeo-17 network. Positive = network activity positively associated with ADHD prediction. Negative = negatively associated.

| Network | Signed (all) | ADHD+ | ADHD- | Direction | n_ROIs |
|---------|:------------:|:-----:|:-----:|:---------:|:------:|
| ContC | +0.000336 | +0.000320 | +0.000348 | +ADHD | 4 |
| VisPeri | +0.000163 | +0.000148 | +0.000174 | +ADHD | 12 |
| SomMotA | +0.000146 | +0.000143 | +0.000147 | +ADHD | 16 |
| DefaultB | +0.000139 | +0.000133 | +0.000144 | +ADHD | 16 |
| SalVentAttnB | +0.000123 | +0.000127 | +0.000121 | +ADHD | 6 |
| VisCent | +0.000114 | +0.000110 | +0.000117 | +ADHD | 10 |
| DorsAttnA | +0.000078 | +0.000078 | +0.000078 | +ADHD | 13 |
| ContB | +0.000057 | +0.000057 | +0.000058 | +ADHD | 8 |
| SalVentAttnA | +0.000044 | +0.000042 | +0.000046 | +ADHD | 15 |
| DefaultA | +0.000005 | +0.000003 | +0.000007 | +ADHD | 21 |
| DefaultC | -0.000001 | -0.000009 | +0.000004 | -ADHD | 10 |
| DorsAttnB | -0.000008 | -0.000005 | -0.000011 | -ADHD | 12 |
| TempPar | -0.000012 | -0.000015 | -0.000010 | -ADHD | 8 |
| SomMotB | -0.000015 | +0.000017 | -0.000038 | -ADHD | 8 |
| Limbic_OFC | -0.000087 | -0.000088 | -0.000086 | -ADHD | 4 |
| ContA | -0.000143 | -0.000143 | -0.000144 | -ADHD | 13 |
| Limbic_TempPole | -0.000201 | -0.000198 | -0.000203 | -ADHD | 4 |

## 4. Network-Level Analysis: Input Projection Weights per Expert

Weight magnitudes of the first linear layer in each expert, grouped by Yeo-17 network. Shows which sub-networks each expert learns to attend to.

### Expert: DMN

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| DefaultC | 0.7003 | 0.1279 | 1.0510 | 10 |
| DefaultA | 0.6987 | 0.1416 | 1.2918 | 21 |
| DefaultB | 0.6844 | 0.0764 | 0.9251 | 16 |
| TempPar | 0.6675 | 0.0558 | 0.7497 | 8 |

**Top 5 ROIs by weight magnitude:**

| ROI | Network | Weight Norm |
|:---:|---------|:----------:|
| 171 | DefaultA | 1.2918 |
| 134 | DefaultC | 1.0510 |
| 126 | DefaultB | 0.9251 |
| 168 | DefaultA | 0.8261 |
| 175 | DefaultB | 0.7512 |

### Expert: Executive

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| ContA | 0.7627 | 0.0725 | 0.8998 | 13 |
| ContC | 0.7241 | 0.0271 | 0.7704 | 4 |
| DorsAttnA | 0.6964 | 0.0733 | 0.8235 | 13 |
| DorsAttnB | 0.6909 | 0.0471 | 0.7886 | 12 |
| ContB | 0.6625 | 0.0250 | 0.7168 | 8 |

**Top 5 ROIs by weight magnitude:**

| ROI | Network | Weight Norm |
|:---:|---------|:----------:|
| 86 | ContA | 0.8998 |
| 92 | ContA | 0.8956 |
| 155 | DorsAttnA | 0.8235 |
| 88 | ContA | 0.8159 |
| 156 | DorsAttnA | 0.8036 |

### Expert: Salience

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| SalVentAttnB | 0.9937 | 0.1690 | 1.3545 | 6 |
| Limbic_OFC | 0.9192 | 0.0501 | 0.9851 | 4 |
| SalVentAttnA | 0.8787 | 0.0655 | 1.0268 | 15 |
| Limbic_TempPole | 0.8628 | 0.0545 | 0.9253 | 4 |

**Top 5 ROIs by weight magnitude:**

| ROI | Network | Weight Norm |
|:---:|---------|:----------:|
| 73 | SalVentAttnB | 1.3545 |
| 71 | SalVentAttnA | 1.0268 |
| 74 | SalVentAttnB | 1.0210 |
| 83 | Limbic_OFC | 0.9851 |
| 161 | SalVentAttnA | 0.9663 |

### Expert: SensoriMotor

| Network | Mean Weight | Std | Max | n_ROIs |
|---------|:----------:|:---:|:---:|:------:|
| VisPeri | 0.8267 | 0.0797 | 0.9709 | 12 |
| SomMotB | 0.7825 | 0.0513 | 0.8936 | 8 |
| SomMotA | 0.7741 | 0.1044 | 1.0058 | 16 |
| VisCent | 0.7537 | 0.0542 | 0.8609 | 10 |

**Top 5 ROIs by weight magnitude:**

| ROI | Network | Weight Norm |
|:---:|---------|:----------:|
| 29 | SomMotA | 1.0058 |
| 10 | VisPeri | 0.9709 |
| 25 | SomMotA | 0.9489 |
| 150 | VisPeri | 0.9478 |
| 17 | VisPeri | 0.9179 |

## 5. ROI-Level Analysis: Top 20 ROIs by Absolute Saliency

| Rank | ROI | Network | Circuit | Saliency | ADHD+ | ADHD- | Diff |
|:----:|:---:|---------|---------|:--------:|:-----:|:-----:|:----:|
| 1 | 171 | DefaultA | DMN | 0.001537 | 0.001504 | 0.001561 | -0.000058 |
| 2 | 73 | SalVentAttnB | Salience | 0.001489 | 0.001519 | 0.001466 | +0.000053 |
| 3 | 134 | DefaultC | DMN | 0.001080 | 0.001042 | 0.001108 | -0.000065 |
| 4 | 29 | SomMotA | SensoriMotor | 0.000866 | 0.000822 | 0.000899 | -0.000077 |
| 5 | 126 | DefaultB | DMN | 0.000837 | 0.000809 | 0.000859 | -0.000050 |
| 6 | 20 | SomMotA | SensoriMotor | 0.000823 | 0.000780 | 0.000854 | -0.000074 |
| 7 | 71 | SalVentAttnA | Salience | 0.000774 | 0.000785 | 0.000765 | +0.000021 |
| 8 | 10 | VisPeri | SensoriMotor | 0.000750 | 0.000732 | 0.000764 | -0.000032 |
| 9 | 21 | SomMotA | SensoriMotor | 0.000708 | 0.000670 | 0.000736 | -0.000066 |
| 10 | 83 | Limbic_OFC | Salience | 0.000681 | 0.000695 | 0.000669 | +0.000026 |
| 11 | 150 | VisPeri | SensoriMotor | 0.000671 | 0.000629 | 0.000702 | -0.000073 |
| 12 | 179 | TempPar | DMN | 0.000664 | 0.000652 | 0.000672 | -0.000021 |
| 13 | 17 | VisPeri | SensoriMotor | 0.000660 | 0.000614 | 0.000696 | -0.000082 |
| 14 | 34 | SomMotB | SensoriMotor | 0.000658 | 0.000635 | 0.000676 | -0.000041 |
| 15 | 168 | DefaultA | DMN | 0.000649 | 0.000627 | 0.000666 | -0.000039 |
| 16 | 32 | SomMotB | SensoriMotor | 0.000646 | 0.000628 | 0.000659 | -0.000030 |
| 17 | 82 | Limbic_OFC | Salience | 0.000632 | 0.000643 | 0.000624 | +0.000020 |
| 18 | 174 | DefaultB | DMN | 0.000625 | 0.000601 | 0.000643 | -0.000042 |
| 19 | 15 | VisPeri | SensoriMotor | 0.000621 | 0.000598 | 0.000637 | -0.000039 |
| 20 | 74 | SalVentAttnB | Salience | 0.000615 | 0.000630 | 0.000605 | +0.000025 |

## 6. ROI-Level Analysis: Largest ADHD+ vs ADHD- Absolute Saliency Differences

ROIs where absolute gradient saliency differs most between classes. Positive diff = more salient for ADHD+.

| Rank | ROI | Network | Circuit | Diff | ADHD+ | ADHD- |
|:----:|:---:|---------|---------|:----:|:-----:|:-----:|
| 1 | 17 | VisPeri | SensoriMotor | -0.000082 | 0.000614 | 0.000696 |
| 2 | 29 | SomMotA | SensoriMotor | -0.000077 | 0.000822 | 0.000899 |
| 3 | 20 | SomMotA | SensoriMotor | -0.000074 | 0.000780 | 0.000854 |
| 4 | 150 | VisPeri | SensoriMotor | -0.000073 | 0.000629 | 0.000702 |
| 5 | 21 | SomMotA | SensoriMotor | -0.000066 | 0.000670 | 0.000736 |
| 6 | 134 | DefaultC | DMN | -0.000065 | 0.001042 | 0.001108 |
| 7 | 171 | DefaultA | DMN | -0.000058 | 0.001504 | 0.001561 |
| 8 | 73 | SalVentAttnB | Salience | +0.000053 | 0.001519 | 0.001466 |
| 9 | 5 | VisCent | SensoriMotor | -0.000052 | 0.000578 | 0.000630 |
| 10 | 25 | SomMotA | SensoriMotor | -0.000050 | 0.000587 | 0.000637 |
| 11 | 126 | DefaultB | DMN | -0.000050 | 0.000809 | 0.000859 |
| 12 | 31 | SomMotA | SensoriMotor | -0.000050 | 0.000431 | 0.000481 |
| 13 | 3 | VisCent | SensoriMotor | -0.000048 | 0.000487 | 0.000535 |
| 14 | 18 | SomMotA | SensoriMotor | -0.000044 | 0.000448 | 0.000491 |
| 15 | 24 | SomMotA | SensoriMotor | -0.000042 | 0.000426 | 0.000468 |
| 16 | 174 | DefaultB | DMN | -0.000042 | 0.000601 | 0.000643 |
| 17 | 34 | SomMotB | SensoriMotor | -0.000041 | 0.000635 | 0.000676 |
| 18 | 105 | ContC | Executive | -0.000040 | 0.000319 | 0.000359 |
| 19 | 39 | SomMotB | SensoriMotor | -0.000040 | 0.000471 | 0.000510 |
| 20 | 14 | VisPeri | SensoriMotor | -0.000040 | 0.000402 | 0.000442 |

## 7. ROI-Level Analysis: Signed Gradient Saliency (Direction)

Signed saliency shows the direction of each ROI's relationship with ADHD. **Positive** = increasing this ROI's signal pushes prediction toward ADHD+. **Negative** = pushes toward ADHD-.

### Top 10 ROIs with Strongest Positive (+ADHD) Relationship

| Rank | ROI | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|---------|---------|:------------:|:------------:|:------------:|
| 1 | 32 | SomMotB | SensoriMotor | +0.000570 | +0.000557 | +0.000580 |
| 2 | 21 | SomMotA | SensoriMotor | +0.000483 | +0.000425 | +0.000527 |
| 3 | 20 | SomMotA | SensoriMotor | +0.000445 | +0.000427 | +0.000458 |
| 4 | 14 | VisPeri | SensoriMotor | +0.000416 | +0.000392 | +0.000434 |
| 5 | 15 | VisPeri | SensoriMotor | +0.000403 | +0.000384 | +0.000418 |
| 6 | 1 | VisCent | SensoriMotor | +0.000390 | +0.000329 | +0.000435 |
| 7 | 103 | ContC | Executive | +0.000377 | +0.000371 | +0.000381 |
| 8 | 173 | DefaultB | DMN | +0.000371 | +0.000376 | +0.000367 |
| 9 | 176 | DefaultC | DMN | +0.000370 | +0.000371 | +0.000370 |
| 10 | 160 | SalVentAttnA | Salience | +0.000368 | +0.000361 | +0.000374 |

### Top 10 ROIs with Strongest Negative (-ADHD) Relationship

| Rank | ROI | Network | Circuit | Signed (all) | Signed ADHD+ | Signed ADHD- |
|:----:|:---:|---------|---------|:------------:|:------------:|:------------:|
| 1 | 92 | ContA | Executive | -0.000477 | -0.000487 | -0.000470 |
| 2 | 91 | ContA | Executive | -0.000341 | -0.000343 | -0.000340 |
| 3 | 171 | DefaultA | DMN | -0.000318 | -0.000350 | -0.000293 |
| 4 | 35 | SomMotB | SensoriMotor | -0.000309 | -0.000250 | -0.000353 |
| 5 | 164 | ContA | Executive | -0.000291 | -0.000296 | -0.000288 |
| 6 | 31 | SomMotA | SensoriMotor | -0.000288 | -0.000220 | -0.000339 |
| 7 | 36 | SomMotB | SensoriMotor | -0.000281 | -0.000229 | -0.000321 |
| 8 | 121 | DefaultA | DMN | -0.000276 | -0.000277 | -0.000276 |
| 9 | 80 | Limbic_TempPole | Salience | -0.000258 | -0.000253 | -0.000263 |
| 10 | 81 | Limbic_TempPole | Salience | -0.000256 | -0.000251 | -0.000259 |
